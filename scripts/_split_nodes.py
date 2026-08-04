"""Migrate ``nodegraph/nodes.py`` into one module per node type (V2.20).

    python scripts/_split_nodes.py --plan          # report the partition, write nothing
    python scripts/_split_nodes.py --apply         # write nodegraph/catalog/** + the facade

A **script**, not a hand edit, because this moves ~7,000 lines across ~63 files and the
failure mode of doing it by hand is a silently dropped socket annotation or a helper that
ends up in two modules and then diverges. A script is reviewable, re-runnable, and the
partition it applies is derived from the code itself rather than from anyone's reading of it.

How the partition is derived
----------------------------
1. Parse the top-level statements of ``nodes.py`` and build the **reference graph** over the
   names they bind (who uses whom).
2. Every ``register_node``/``define_node`` call is a node **anchor**. Walk the graph from each
   anchor to get its reachable set.
3. A definition reachable from **exactly one** anchor is that node's **private** code and
   moves into the node's module. Reachable from **two or more** ⇒ **shared**, and its target
   module comes from :data:`SHARED` — a hand-authored grouping, because "which concern does
   this helper belong to" is a judgement the graph cannot make, and getting it right is what
   decides whether a helper edit re-keys 4 nodes or 63.
4. Anything else is listed in :data:`ORPHANS` with an explicit destination. The script
   **refuses to run** if it meets a statement it cannot place, so a new helper added to
   ``nodes.py`` before the migration cannot be silently dropped.

Imports are **synthesized per module**, never copied: for each emitted module the script
collects the free names its statements actually use and resolves each one to the import (or
sibling catalog module) that provides it, narrowing ``from x import a, b, c`` down to the
names that module needs. That narrowing is not tidiness — the per-node fingerprint is a
digest over each module's transitive import closure, so a spurious import is a spurious
dependency that re-keys the node whenever an unrelated file changes.

Text handling: each statement carries the source lines from the end of the previous statement
through its own end, so the comment blocks and section rules that sit *above* a definition
travel with it. Line endings are preserved (this file is CRLF).
"""
from __future__ import annotations

import ast
import builtins
import collections
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES_PY = os.path.join(REPO, "nodegraph", "nodes.py")
CATALOG = os.path.join(REPO, "nodegraph", "catalog")

#: Shared helper -> its module under ``nodegraph/catalog/_shared/``, grouped BY CONCERN.
#:
#: This table is the single most consequential thing in the file, because the per-node
#: fingerprint is a digest over each module's transitive import closure: a helper's blast
#: radius is *its module's* importer set, not its own user set. Grouping all 34 helpers into
#: one prelude would re-key 1497 node-keys per helper edit (measured); this 16-module grouping
#: re-keys 76, against a theoretical floor of 21 for one module per distinct user set — 95% of
#: the available granularity, and in every module the importer count equals its widest
#: member's own user count plus 0 or 1, so the grouping is never itself the bottleneck.
#:
#: Four helpers are irreducible and no grouping helps: ``to_pixels_v2`` (18 users),
#: ``_DIM_KAX`` (17), ``_FIELD_TYPES``/``_map_image`` (14) and ``SAMPLING_KEY`` (14). Three of
#: those are a pure function and two frozen one-liners that will not be edited. ``_map_image``
#: is the real cost: 91 lines of tiling/halo/GPU policy that WILL be edited, re-keying 22% of
#: the catalog each time. Hoisting it into the existing ``nodegraph/streaming.py`` was measured
#: and is worse (14 -> 26 nodes), because 11 nodes use that module without using ``_map_image``.
#:
#: The rule the grouping follows: **a helper's home is the smallest module whose entire user
#: set wants to be re-keyed together.**
SHARED: Dict[str, Tuple[str, ...]] = {
    # enumerate a Dataset's acquisition units, with and without per-node progress
    "planes": ("_each_plane", "_each_plane_p", "_each_volume_p"),
    # progress adapters for computes whose unit is not one plane
    "progress": ("_parallel_progress", "_UnitBar"),
    # physical value -> pixels/frames. A true leaf: no first-party imports at all, which is
    # why its 18 importers are harmless — only editing this one pure function can re-key them.
    "units": ("to_pixels_v2",),
    # a µm structuring-element radius, its 3D axial twin, and the 1-voxel-window refusal
    "kernel_radius": ("_win", "_require_window", "_radius_px", "_radius_z_px",
                      "_AXIAL_TWIN_DOC", "_InRadius"),
    # the declared full-scale intensity (2**bit_depth - 1)
    "full_scale": ("_declared_full_scale",),
    # the lazy tile/plane/volume apply path + the kernel-param Field gate that forces it
    # off tiles. Kept OUT of nodegraph/streaming.py — see the note above.
    "map_image": ("_FIELD_TYPES", "_kernel_field_varies", "_map_image"),
    # the per-dim data-access footprint maps
    "dim_footprint": ("_DIM_GRAN", "_DIM_KAX", "_DIM_GRAN_GLOBAL"),
    # the sampling-grid provenance stamp (§7b), read and write
    "sampling": ("SAMPLING_KEY", "_sampled", "_sampling_of"),
    # the "segment on enhanced, measure on raw" seam and its geometry/provenance guard
    "raw_measure": ("_InRaw", "_intensity_provider"),
    # validated access to a Label instance's raster + per-id centroid/count reduction
    "labels": ("_label_raster", "_label_centroids"),
    # the object-table schema, per-track velocity, and the frame-interval socket/reader
    "objects": ("_object_table", "_object_velocity", "_frame_interval_s",
                "_InFrameInterval"),
    "dvc": ("_dvc_rows",),
    "drift_layers": ("_layers_drift",),
    "global_scalar": ("_global_scalar",),
    "ndimage": ("_ndi",),
    # The zone/group boundary compute: a two-line identity pass-through that SIX ops share
    # across two family modules. It lives here rather than being copied into both so the two
    # cannot drift — and so no two catalog modules define a function of the same name, which
    # `selftest._param_key_index` requires (its tables are keyed by function name).
    "passthrough": ("_compute_zone_passthrough",),
}

#: Names whose destination is decided by hand, overriding whatever the reference graph
#: concluded. Each entry is a case where the graph is right about the *references* and wrong
#: about the *unit of change*.
FORCE: Dict[str, str] = {
    # Reference implementations with no caller at all: `_rl` is the definition that
    # `_rl_gaussian`'s fast path is validated against, and `gaussian_psf` is the explicit-PSF
    # form behind `diffraction_sigmas`. They belong WITH the node they document, not in a
    # shared module — a `_shared` home would give them a blast radius they cannot earn, and
    # separating a reference implementation from the fast path it certifies is how the two
    # come to disagree.
    "gaussian_psf": "enhance.deconvolve",
    "_rl": "enhance.deconvolve",
    # The Iterate variable-slot machinery must be ONE module with its registration, because
    # it is module-level EXECUTION, not just definitions: `_ITER_IN/_ITER_OUT/_ITER_MODES`
    # are built by a `for` loop that APPENDS. Split the accumulators from the loop and every
    # live reload of the loop's module appends another full set of slots to lists that were
    # never re-initialised — 24 -> 48 -> 72 sockets, silently, because `define_node` does not
    # deduplicate socket names.
    "MAX_VARS": "flow.iterate",
    "_iterate_slot_sockets": "flow.iterate",
    "_ITER_IN": "flow.iterate",
    "_ITER_OUT": "flow.iterate",
    "_ITER_MODES": "flow.iterate",
    # Promoted to _shared despite having exactly ONE user: it is the literal volume twin of
    # `_each_plane_p` ("As _each_plane_p, but over (m, t, c) volumes") and the two must stay
    # in sync or the two-level progress report diverges between plane- and volume-unit
    # computes. Costs +9 over-invalidation, the largest single item of slack in the design,
    # and accepted deliberately: the alternative is a private copy of a helper whose sibling
    # is shared, which the next volume-unit node either duplicates or promotes anyway.
    "_each_volume_p": "_shared.planes",
}

#: Statements to DROP, each with the reason it is not lost. Keyed by the name they bind;
#: unnamed statements are handled by kind in :func:`plan`.
DROP: Dict[str, str] = {
    # The facade defines its own __all__ over the 24-name compatibility surface; the old
    # 6-name list was already not the contract (SAMPLING_KEY has 8 importers and was absent).
    "__all__": "the facade declares its own, over the real compatibility surface",
}

#: op_keys registered from ONE site (a ``for`` loop over one shared compute) -> their module.
#: Their op_key cannot name the module because several share it.
FAMILIES: Dict[str, str] = {
    "zone.repeat_in": "zone.boundary", "zone.repeat_out": "zone.boundary",
    "zone.sim_in": "zone.boundary", "zone.sim_out": "zone.boundary",
    "group.input": "group.interface", "group.output": "group.interface",
}

#: Names the facade ``nodegraph/nodes.py`` must re-export. Computed by AST-walking every
#: first-party file for ``from nodegraph.nodes import X``, ``import nodegraph.nodes as NN``
#: + ``NN.<attr>``, and ``from nodegraph import X`` where ``nodegraph/__init__`` sourced X
#: here — then pasted in, so the list is auditable rather than re-derived at build time.
#: Seven are public API (the first five also reached through ``nodegraph/__init__``'s
#: ``__all__``); the rest are private helpers that ``nodegraph/selftest.py`` and
#: ``scripts/_nodelab_v2_phase5_probe.py`` pin directly. They are part of the contract in
#: practice: dropping one turns a verification gate into an ImportError before it can report
#: anything about the split.
FACADE: Tuple[str, ...] = (
    "COMPUTES", "OVERLAY_KEY", "SAMPLING_KEY", "_DIM_KAX", "_MEASURE_COLUMNS",
    "_MEASURE_SHAPE", "_OBJECT_FIELDS", "_OBJECT_METRIC_COLUMNS", "_SEGMENT_2D_ONLY",
    "_exp_fit_t", "_gauss_blur_zero", "_label_centroids", "_map_image", "_measure_stats",
    "_rl", "_rl_gaussian", "_rolling_mean_t", "_snap_bit_depth", "_temporal_gain_field",
    "_tile_means", "diffraction_sigmas", "gaussian_psf", "register_node", "to_pixels_v2",
)


# ── source-level statement model ─────────────────────────────────────────────

class Stmt:
    """One top-level statement plus the source text that belongs to it."""

    __slots__ = ("node", "index", "text", "binds", "refs")

    def __init__(self, node: ast.stmt, index: int, text: str) -> None:
        self.node = node
        self.index = index
        self.text = text
        self.binds: Tuple[str, ...] = tuple(_binds(node))
        self.refs: Set[str] = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        self.refs |= _attr_roots(node)

    @property
    def kind(self) -> str:
        return type(self.node).__name__

    def __repr__(self) -> str:
        return f"<Stmt {self.index} {self.kind} {','.join(self.binds) or '-'}>"


def _binds(st: ast.stmt) -> List[str]:
    out: List[str] = []
    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        out.append(st.name)
    elif isinstance(st, ast.Assign):
        for t in st.targets:
            out += [n.id for n in ast.walk(t) if isinstance(n, ast.Name)]
    elif isinstance(st, ast.AnnAssign):
        if isinstance(st.target, ast.Name):
            out.append(st.target.id)
    elif isinstance(st, (ast.Import, ast.ImportFrom)):
        for a in st.names:
            out.append(a.asname or a.name.split(".")[0])
    elif isinstance(st, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
        # a compound statement can bind its loop/target variables at module level
        for n in ast.walk(st):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.append(n.id)
    return out


def _attr_roots(st: ast.stmt) -> Set[str]:
    """Roots of dotted references (``np.zeros`` -> ``np``) — already covered by Name nodes,
    kept as a named step because a missed reference silently drops an import."""
    return {n.value.id for n in ast.walk(st)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}


def read_statements(path: str) -> Tuple[List[Stmt], str, str]:
    """Top-level statements with their leading comments attached, plus the raw source."""
    with open(path, encoding="utf-8", newline="") as fh:
        raw = fh.read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(nl)
    tree = ast.parse(raw)
    stmts: List[Stmt] = []
    prev_end = 0                                   # 0-based index of first unclaimed line
    for i, node in enumerate(tree.body):
        start = node.lineno - 1
        # a decorated def's lineno points at `def`, not the decorator
        if getattr(node, "decorator_list", None):
            start = min(start, min(d.lineno for d in node.decorator_list) - 1)
        end = node.end_lineno                      # 1-based, inclusive
        text = nl.join(lines[prev_end:end])
        prev_end = end
        stmts.append(Stmt(node, i, text))
    trailing = nl.join(lines[prev_end:])
    return stmts, nl, trailing


# ── partition ────────────────────────────────────────────────────────────────

class Partition:
    def __init__(self, stmts: Sequence[Stmt]) -> None:
        self.stmts = list(stmts)
        self.binder: Dict[str, int] = {}
        for s in self.stmts:
            for nm in s.binds:
                self.binder[nm] = s.index
        self.edges: Dict[int, Set[int]] = {
            s.index: {self.binder[n] for n in s.refs
                      if n in self.binder and self.binder[n] != s.index}
            for s in self.stmts
        }
        self.anchors = self._anchors()
        self.imports = [s for s in self.stmts
                        if isinstance(s.node, (ast.Import, ast.ImportFrom))]

    def _anchors(self) -> List[Tuple[Stmt, Tuple[str, ...], Optional[str]]]:
        """(statement, op_keys, compute name) for every registration site, including the
        ``for``-loop sites that register a family from one compute."""
        out = []
        for s in self.stmts:
            ops, compute = [], None
            for call in [n for n in ast.walk(s.node) if isinstance(n, ast.Call)]:
                fname = getattr(call.func, "id", getattr(call.func, "attr", ""))
                if fname not in ("register_node", "define_node"):
                    continue
                for kw in call.keywords:
                    if kw.arg == "op_key" and isinstance(kw.value, ast.Constant):
                        ops.append(kw.value.value)
                if call.args and isinstance(call.args[0], ast.Name):
                    compute = call.args[0].id
            # a for-loop family: op_keys are the loop's literal tuple, not a keyword
            if not ops and isinstance(s.node, ast.For):
                if any(isinstance(n, ast.Call) and
                       getattr(n.func, "id", "") in ("register_node", "define_node")
                       for n in ast.walk(s.node)):
                    ops = [c.value for c in ast.walk(s.node)
                           if isinstance(c, ast.Constant) and isinstance(c.value, str)
                           and "." in c.value and " " not in c.value]
            if ops:
                out.append((s, tuple(dict.fromkeys(ops)), compute))
        return out

    def reach(self, start: int) -> Set[int]:
        seen, stack = set(), [start]
        while stack:
            k = stack.pop()
            for j in self.edges[k]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        return seen

    def owners(self) -> Dict[int, Set[str]]:
        """statement index -> the set of anchor keys that can reach it."""
        out: Dict[int, Set[str]] = collections.defaultdict(set)
        for s, ops, _ in self.anchors:
            key = ops[0]
            for j in self.reach(s.index):
                out[j].add(key)
        return out


def module_for(op_key: str) -> str:
    """``"enhance.gamma"`` -> ``"enhance.gamma"`` (module path under the catalog).

    The op_key IS the module path: mechanical, so a mismatch between what a module is called
    and what it registers is impossible to introduce by accident. Families override it."""
    if op_key in FAMILIES:
        return FAMILIES[op_key]
    return op_key


def plan(part: Partition) -> dict:
    owners = part.owners()
    anchor_idx = {s.index for s, _, _ in part.anchors}
    import_idx = {s.index for s in part.imports}
    assign: Dict[int, str] = {}                # statement index -> target module
    unplaced: List[Stmt] = []

    for s, ops, _ in part.anchors:
        assign[s.index] = module_for(ops[0])

    n_assert = sum(1 for s in part.stmts if isinstance(s.node, ast.Assert))
    assert n_assert <= 1, (f"{n_assert} top-level asserts — the placement below assumes the "
                           f"only one is the MAX_VARS/iterate consistency check")
    for s in part.stmts:
        if s.index in anchor_idx or s.index in import_idx:
            continue
        # ── statements that bind no name: placed by kind, each for a stated reason ──
        if not s.binds:
            if s.index == 0 and isinstance(s.node, ast.Expr):
                continue          # the module docstring — the facade gets its own, and the
                                  # contract/conventions prose moves to catalog/__init__
            if isinstance(s.node, ast.Assert):
                assign[s.index] = "flow.iterate"     # the MAX_VARS agreement check: it guards
                continue                             # the slot machinery, so it travels with it
            if isinstance(s.node, ast.Expr) and isinstance(s.node.value, ast.Call):
                continue          # `_catalog.load()` — the facade keeps the registration seam
            unplaced.append(s)
            continue
        forced = next((FORCE[nm] for nm in s.binds if nm in FORCE), None)
        if forced is not None:
            assign[s.index] = forced
            continue
        if any(nm in DROP for nm in s.binds):
            continue
        who = owners.get(s.index, set())
        shared = next((mod for nm in s.binds for mod, names in SHARED.items()
                       if nm in names), None)
        if shared is not None:
            assign[s.index] = f"_shared.{shared}"
        elif len(who) == 1:
            assign[s.index] = module_for(next(iter(who)))
        else:
            unplaced.append(s)
    return {"assign": assign, "unplaced": unplaced, "owners": owners}


# ── import synthesis ─────────────────────────────────────────────────────────

class ImportResolver:
    """Resolves a free name to the import line (or catalog module) that provides it."""

    def __init__(self, part: Partition, assign: Dict[int, str]) -> None:
        self.from_imports: Dict[str, Tuple[str, str]] = {}   # local -> (module, orig name)
        self.plain_imports: Dict[str, str] = {}              # local -> full import source
        for s in part.imports:
            node = s.node
            if isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    continue
                for a in node.names:
                    local = a.asname or a.name
                    # `from nodegraph import gpu as _gpu` must become
                    # `import nodegraph.gpu as _gpu`. The two are equivalent for the code that
                    # uses `_gpu.ndimage(...)`, but NOT for the dependency graph: the `from`
                    # form resolves through `nodegraph/__init__`, which imports the facade,
                    # which loads the entire catalog — so it is both an import cycle from
                    # inside a node module and a closure that contains every node in the
                    # catalog. `test_catalog_import_hygiene` rejects it (rule 2); it caught
                    # this exact line surviving into `_shared/ndimage.py`.
                    if node.module == "nodegraph" and os.path.exists(
                            os.path.join(REPO, "nodegraph", a.name + ".py")):
                        self.plain_imports[local] = (
                            f"import nodegraph.{a.name} as {local}" if local != a.name
                            else f"import nodegraph.{a.name}")
                        continue
                    self.from_imports[local] = (node.module or "", a.name)
            else:
                for a in node.names:
                    local = a.asname or a.name.split(".")[0]
                    self.plain_imports[local] = (f"import {a.name} as {a.asname}"
                                                 if a.asname else f"import {a.name}")
        self.provider: Dict[str, str] = {}                   # name -> target module
        for s in part.stmts:
            mod = assign.get(s.index)
            if mod is None:
                continue
            for nm in s.binds:
                self.provider[nm] = mod

    def lines(self, module: str, needed: Set[str], nl: str) -> str:
        """The import block for ``module``, given the free names it uses."""
        std: Dict[str, List[str]] = collections.defaultdict(list)
        plain: List[str] = []
        catalog: Dict[str, List[str]] = collections.defaultdict(list)
        for nm in sorted(needed):
            if nm in ("COMPUTES", "register_node"):
                catalog["nodegraph.catalog._base"].append(nm)
            elif nm in self.provider and self.provider[nm] != module:
                catalog[f"nodegraph.catalog.{self.provider[nm]}"].append(nm)
            elif nm in self.from_imports:
                src, orig = self.from_imports[nm]
                std[src].append(f"{orig} as {nm}" if orig != nm else nm)
            elif nm in self.plain_imports:
                plain.append(self.plain_imports[nm])
        out = ["from __future__ import annotations", ""]
        out += sorted(set(plain))
        stdlib = {m: v for m, v in std.items() if not m.startswith("nodegraph")}
        first = {m: v for m, v in std.items() if m.startswith("nodegraph")}
        for group in (stdlib, first, catalog):
            if group:
                out.append("")
            for mod in sorted(group):
                names = ", ".join(sorted(set(group[mod])))
                line = f"from {mod} import {names}"
                out.append(line if len(line) <= 96 else
                           f"from {mod} import (\n    " + ",\n    ".join(
                               sorted(set(group[mod]))) + ",\n)")
        return nl.join(out)


def _scope_free(node: ast.AST) -> Set[str]:
    """Names ``node`` needs from MODULE scope — a real free-variable analysis.

    Not "every ``ast.Name`` in the subtree", which is what a first cut does and which is
    wrong in a way that quietly corrupts the whole migration: a *function-local* variable
    that happens to share a name with some module-level binding elsewhere looks like a
    reference to it, so the emitter synthesizes an import for it. That actually happened —
    every compute with a ``for _m in range(ax.m)`` position loop resolved ``_m`` to the
    ``_iterate_slot_sockets`` unpacking loop's temporary and emitted
    ``from nodegraph.catalog.flow.iterate import _m``. The import even *works* (the name
    exists), so nothing fails; it just fabricates a dependency, which pollutes the module's
    fingerprint closure and — because importing a module executes it — silently reorders
    catalog registration. The snapshot gate caught it as an order change 29 positions in.

    Python's actual rule: a name assigned anywhere in a function body is local to that
    function. So per scope, ``free = loaded - bound``, recursing into nested scopes with the
    enclosing scope's bindings removed."""
    bound: Set[str] = set()
    loaded: Set[str] = set()
    nested: List[ast.AST] = []

    def visit(n: ast.AST, top: bool) -> None:
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(child.name)
                # a decorator/default is evaluated in the ENCLOSING scope
                for d in getattr(child, "decorator_list", []):
                    visit_expr(d)
                args = getattr(child, "args", None)
                if args is not None:
                    for d in list(args.defaults) + [x for x in args.kw_defaults if x]:
                        visit_expr(d)
                nested.append(child)
                continue
            if isinstance(child, ast.Name):
                (bound if isinstance(child.ctx, (ast.Store, ast.Del)) else loaded).add(child.id)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for a in child.names:
                    bound.add(a.asname or a.name.split(".")[0])
            elif isinstance(child, ast.ExceptHandler) and child.name:
                bound.add(child.name)
            elif isinstance(child, ast.Global):
                bound.update(child.names)
            visit(child, False)

    def visit_expr(n: ast.AST) -> None:
        for x in ast.walk(n):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load):
                loaded.add(x.id)

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = node.args
        for arg in (a.posonlyargs + a.args + a.kwonlyargs
                    + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else [])):
            bound.add(arg.arg)
    visit(node, True)
    free = loaded - bound
    for child in nested:
        free |= _scope_free(child) - bound
    return free


def free_names(stmts: Sequence[Stmt], own: Set[str]) -> Set[str]:
    """Names these statements reference but do not themselves define."""
    used: Set[str] = set()
    for s in stmts:
        used |= _scope_free(s.node)
    return {n for n in used - own if not hasattr(builtins, n)}


# ── report ───────────────────────────────────────────────────────────────────

def report(part: Partition, p: dict) -> int:
    assign = p["assign"]
    by_mod: Dict[str, List[Stmt]] = collections.defaultdict(list)
    for s in part.stmts:
        if s.index in assign:
            by_mod[assign[s.index]].append(s)
    print(f"{len(part.anchors)} registration sites -> "
          f"{len({m for m in by_mod if not m.startswith('_shared')})} node modules")
    print(f"{len([m for m in by_mod if m.startswith('_shared')])} shared modules")
    lines = lambda ss: sum(s.text.count("\n") + 1 for s in ss)
    print(f"\n{'module':46s} {'stmts':>6s} {'lines':>7s}")
    for mod in sorted(by_mod):
        print(f"  {mod:44s} {len(by_mod[mod]):6d} {lines(by_mod[mod]):7d}")
    if p["unplaced"]:
        print(f"\nREFUSING: {len(p['unplaced'])} statement(s) have no destination.")
        print("Add each to SHARED (if used by 2+ nodes) or ORPHANS (with a reason):")
        for s in p["unplaced"]:
            who = p["owners"].get(s.index, set())
            print(f"  {s.kind:12s} L{s.node.lineno:6d} {','.join(s.binds) or '-':32s} "
                  f"used by {len(who)} node(s)")
        return 1
    return 0


# ── emit ─────────────────────────────────────────────────────────────────────

def _headline(part: Partition, mod: str, stmts: Sequence[Stmt]) -> str:
    """A real module docstring for a node module, built from its own registration call:
    ``Gamma (``enhance.gamma``) — <first sentence of the node's description>.``

    Generated rather than stubbed because a file called ``gamma.py`` whose docstring says
    "TODO" is worse than no docstring: the label and the one-line description are already
    written in the registration call, and lifting them keeps the two from drifting."""
    label, desc, ops = "", "", []
    for s in stmts:
        for call in [n for n in ast.walk(s.node) if isinstance(n, ast.Call)]:
            if getattr(call.func, "id", "") not in ("register_node", "define_node"):
                continue
            for kw in call.keywords:
                if kw.arg == "op_key" and isinstance(kw.value, ast.Constant):
                    ops.append(kw.value.value)
                if kw.arg == "label" and isinstance(kw.value, ast.Constant) and not label:
                    label = kw.value.value
                if kw.arg == "description" and not desc:
                    try:
                        desc = ast.literal_eval(kw.value)
                    except (ValueError, SyntaxError):
                        desc = ""
    keys = ", ".join(f"``{o}``" for o in dict.fromkeys(ops)) or f"``{mod}``"
    first = _one_sentence(desc)
    head = f"{label or mod} ({keys})"
    return f'"""{head}{" — " + first if first else "."}"""'


def _one_sentence(desc: str) -> str:
    """The first sentence of a node's description, safe to put in a docstring.

    Truncation needs care rather than a slice: cutting mid-way through the catalog's prose
    routinely lands inside a ``` `` ``` span or a parenthesis, and an unbalanced backtick in a
    docstring is a rendering defect that will sit in 64 files. So the cut is taken at a
    sentence end where possible, and a forced cut backs off to a point where every delimiter
    it opened is closed again."""
    text = " ".join((desc or "").split())
    if not text:
        return ""
    for end in (". ", "; "):
        if end in text[:220]:
            text = text[:text.index(end) + 1]
            break
    if len(text) > 200:
        text = text[:197].rsplit(" ", 1)[0] + "…"
    while text.count("`") % 2 or text.count("(") != text.count(")"):
        cut = max(text.rfind(" "), 0)
        if not cut:
            return ""
        text = text[:cut].rstrip(" ,;(")
        if not text.endswith(("…", ".")):
            text += "…"
    return text if text.endswith((".", "…")) else text + "."


def _strip_leading_section(text: str, nl: str) -> str:
    """Drop a leading blank/comment run when it contains a ``# ── … ──`` section divider.

    Only the run *before the first code line*, and only when a divider is in it — so a comment
    written about the definition itself survives, while a divider inherited from the old file's
    layout does not."""
    lines = text.split(nl)
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        i += 1
    head, rest = lines[:i], lines[i:]
    if not rest or not any("─" in h for h in head):
        return text
    return nl.join(rest)


def _emit_module(path: str, body: str, nl: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(body if body.endswith(nl) else body + nl)


def _package_inits(mods: Sequence[str], nl: str) -> None:
    """A minimal ``__init__.py`` for each category subpackage (``enhance/``, ``analysis/``…).

    Deliberately EMPTY of imports: a category package that imported its children would
    re-register the whole category the moment anything touched one node in it, which is the
    same order-destroying trap that ``catalog/__init__`` avoids by not auto-loading."""
    pkgs = {m.rsplit(".", 1)[0] for m in mods if "." in m}
    for pkg in sorted(pkgs):
        path = os.path.join(CATALOG, pkg.replace(".", os.sep), "__init__.py")
        if os.path.exists(path):
            continue
        name = pkg.rsplit(".", 1)[-1]
        _emit_module(path, f'"""Catalog nodes in the ``{name}`` category '
                           f'(one module per node type)."""{nl}', nl)


def apply_split(part: Partition, p: dict, nl: str, order: Sequence[str]) -> int:
    assign = p["assign"]
    by_mod: Dict[str, List[Stmt]] = collections.defaultdict(list)
    for s in part.stmts:
        if s.index in assign:
            by_mod[assign[s.index]].append(s)
    for stmts in by_mod.values():
        stmts.sort(key=lambda s: s.index)          # keep original in-file order
    resolver = ImportResolver(part, assign)

    written = []
    for mod, stmts in sorted(by_mod.items()):
        own = {nm for s in stmts for nm in s.binds}
        needed = free_names(stmts, own)
        is_shared = mod.startswith("_shared.")
        doc = (f'"""{mod.rsplit(".", 1)[-1].replace("_", " ")} — shared catalog helpers."""'
               if is_shared else _headline(part, mod, stmts))
        parts = [doc, "", resolver.lines(mod, needed, nl), ""]
        bodies = [s.text.strip(nl) for s in stmts]
        if is_shared and bodies:
            # A leading comment run that carries a SECTION DIVIDER described a group of
            # definitions in the old single file, not the helper that happened to follow it —
            # `_shared/kernel_radius.py` inherited "── ported catalog (Phase 3) ──" and a
            # paragraph about a batch of enhancement nodes, which is actively misleading at the
            # top of a units helper. Node modules keep theirs: a divider there almost always
            # names the node itself ("── Deconvolve (the two-mode … flagship) ──").
            bodies[0] = _strip_leading_section(bodies[0], nl)
        parts += bodies
        path = os.path.join(CATALOG, mod.replace(".", os.sep) + ".py")
        _emit_module(path, nl.join(parts) + nl, nl)
        written.append(mod)

    node_mods = [m for m in written if not m.startswith("_shared.")]
    _package_inits(node_mods, nl)

    # Modules already on disk from an earlier pass (the pilot extraction) count too: they are
    # no longer in nodes.py, so this run cannot see them as anchors, and rebuilding MODULES
    # from this run alone would silently DE-REGISTER them.
    existing = []
    for dirpath, _dirs, files in os.walk(CATALOG):
        rel = os.path.relpath(dirpath, CATALOG)
        if rel.startswith("_shared") or rel.startswith("__pycache__"):
            continue
        for f in sorted(files):
            if f.endswith(".py") and f != "__init__.py":
                dotted = ((rel.replace(os.sep, ".") + "." if rel != "." else "") + f[:-3])
                if not dotted.startswith("_"):
                    existing.append(dotted)
    known = set(node_mods) | set(existing)

    # MODULES in the ORIGINAL registration order: take the recorded op_key order and map each
    # to its module, keeping first appearance. Order is observable (the link-search menu
    # enumerates the registry), so it is reproduced rather than sorted.
    seq: List[str] = []
    for op in order:
        mod = module_for(op)
        if mod in known and mod not in seq:
            seq.append(mod)
    missing = sorted(known - set(seq))
    if missing:
        print(f"WARNING: {len(missing)} module(s) absent from the recorded order, appended "
              f"(their registration position is a GUESS — check the snapshot): {missing}")
        seq += missing
    n = _write_catalog_init(seq, nl)
    _write_facade(resolver, nl)
    return n, written


FACADE_DOC = '''"""The node catalog's public surface — a facade over :mod:`nodegraph.catalog`.

Importing this module **registers the whole catalog** (that is what
:func:`nodegraph.catalog.load` does, and several callers import this module purely for that
side effect — see ``nodelab_v2/window.py``). The node definitions themselves live one per
module under ``nodegraph/catalog/``, so that a single node can be edited and reloaded into a
running session without re-keying every other node's memoized results
(:mod:`nodegraph.hotreload`).

Everything below is a **re-export**. The names are the compatibility surface this module has
always had: ``COMPUTES``, ``register_node``, ``to_pixels_v2``, ``diffraction_sigmas``,
``gaussian_psf`` and ``OVERLAY_KEY`` are public API (the first five are also re-exported by
``nodegraph/__init__``); the rest are internals that ``nodegraph/selftest.py`` and the
GUI probes pin directly, kept reachable so a verification gate reports on the split instead
of failing to import.

**A caveat for anything that reasons about a compute's origin.** A re-export makes a name
reachable; it does not change where the object was defined. A compute's ``__module__`` is now
its own node module, and this module's AST contains no ``FunctionDef`` at all — so a check
written as ``fn.__module__ == "nodegraph.nodes"``, or one that parses
``inspect.getsource(nodegraph.nodes)``, does not merely break: it silently matches nothing and
reports success over an empty set. Ask :func:`nodegraph.hotreload.is_catalog_op` (registration
provenance) instead, and see ``nodegraph.selftest._catalog_modules`` for the source-level case.

Qt-free.
"""'''


def _write_facade(resolver: "ImportResolver", nl: str) -> None:
    """Replace ``nodegraph/nodes.py`` with the thin re-export facade."""
    groups: Dict[str, List[str]] = collections.defaultdict(list)
    unresolved = []
    for name in FACADE:
        if name in ("COMPUTES", "register_node"):
            groups["nodegraph.catalog._base"].append(name)
        elif name in resolver.provider:
            groups[f"nodegraph.catalog.{resolver.provider[name]}"].append(name)
        else:
            unresolved.append(name)
    if unresolved:
        raise SystemExit(f"REFUSING: {len(unresolved)} facade name(s) were not emitted "
                         f"anywhere and would become an ImportError: {unresolved}")
    lines = [FACADE_DOC, "from __future__ import annotations", "",
             "import nodegraph.catalog as _catalog", "",
             "# Registers every node module, in the order `nodegraph.catalog.MODULES` fixes.",
             "#",
             "# Called HERE rather than in the catalog package's own __init__ so that importing",
             "# any one node module (or `_base`, which they all need) cannot trigger the whole",
             "# catalog at an arbitrary moment.",
             "#",
             "# And called BEFORE the re-exports below, which is load-bearing rather than",
             "# stylistic: `from nodegraph.catalog.analysis.histogram_threshold import"
             " _snap_bit_depth`",
             "# EXECUTES that module, so with the re-exports first, the handful of nodes this",
             "# facade happens to borrow a helper from would register ahead of everything else."
             "",
             "# Registration order is observable (the link-drag search menu enumerates the",
             "# registry) and `scripts/_catalog_snapshot.py` gates it — it caught exactly this.",
             "# With load() first, every module is already in sys.modules and the re-exports",
             "# are pure attribute lookups.",
             "_catalog.load()",
             ""]
    for mod in sorted(groups):
        names = sorted(groups[mod])
        one = f"from {mod} import {', '.join(names)}  # noqa: E402"
        lines.append(one if len(one) <= 96 else
                     f"from {mod} import (  # noqa: E402,F401" + nl + "    "
                     + ("," + nl + "    ").join(names) + "," + nl + ")")
    lines += [
        "",
        "__all__ = [",
    ]
    row = "    "
    for name in FACADE:
        piece = f'"{name}", '
        if len(row) + len(piece) > 96:
            lines.append(row.rstrip())
            row = "    "
        row += piece
    lines.append(row.rstrip().rstrip(","))
    lines.append("]")
    with open(NODES_PY, "w", encoding="utf-8", newline="") as fh:
        fh.write(nl.join(lines) + nl)


def _write_catalog_init(seq: Sequence[str], nl: str) -> int:
    path = os.path.join(CATALOG, "__init__.py")
    with open(path, encoding="utf-8", newline="") as fh:
        cur = fh.read()
    start = cur.index("MODULES: Tuple[str, ...] = (")
    end = cur.index(")", start)
    body = "".join(f'    "{m}",{nl}' for m in seq)
    new = cur[:start] + "MODULES: Tuple[str, ...] = (" + nl + body + cur[end:]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(new)
    return len(seq)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--plan"
    stmts, nl, trailing = read_statements(NODES_PY)
    part = Partition(stmts)
    if not part.anchors:
        # The split has already been applied: nodes.py is the facade and registers nothing.
        # This is a REFUSAL rather than a no-op because `--apply` would otherwise rebuild
        # `catalog/__init__.py`'s MODULES from the zero modules it emitted this run — wiping
        # the list that is the catalog's registration order and de-registering all 65 modules.
        print("nodegraph/nodes.py registers nothing — the split is already applied.\n"
              "Nothing to do. (Refusing to continue: --apply rebuilds catalog/__init__.py's\n"
              "MODULES from what it emits, so running it now would empty that list.)\n\n"
              "To re-derive the split, restore the pre-split nodes.py first.")
        return 0
    p = plan(part)
    rc = report(part, p)
    if rc != 0 or mode != "--apply":
        return rc
    if not FACADE:
        print("\nREFUSING --apply: FACADE is empty. The compatibility surface must be "
              "explicit — nodegraph/selftest.py alone imports 16 names from nodes.py.")
        return 1
    base = os.path.join(REPO, "scripts", "catalog_baseline.json")
    order = json.load(open(base, encoding="utf-8"))["order"] if os.path.exists(base) else []
    if not order:
        print("\nREFUSING --apply: no catalog_baseline.json — registration order is "
              "observable and must be reproduced, not guessed. Run:\n"
              "  python scripts/_catalog_snapshot.py save")
        return 1
    n, written = apply_split(part, p, nl, order)
    print(f"\nwrote {len(written)} modules; catalog/__init__.py lists {n} in order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
