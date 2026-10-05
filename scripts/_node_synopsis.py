#!/usr/bin/env python
"""Node synopsis: one JSON document describing every shipped node - what it does, its inputs
and outputs with their data types, units, defaults and hover prose, its modes, its
dimensionality, its data-access footprint, and the attribute domains it reads and writes.

    python scripts/_node_synopsis.py            # check: exit 1 if codemap/node_synopsis.json is stale
    python scripts/_node_synopsis.py write      # regenerate codemap/node_synopsis.json
    python scripts/_node_synopsis.py show <op>  # print one node's record

Built from the LIVE registry, never from prose: a spec is assembled by factory calls
(``InFloat``, ``DimMode``) and sometimes in a loop, so the only honest source for "what
sockets does this node have" is ``nodegraph.registry.NODES`` after the catalog has imported.
The three GUI-layer ops (``io.load``, ``io.dock``, ``view.viewer``) are registered here by
calling ``nodelab_v2.ops.ensure_ops()``; without PySide6 that step is skipped and the file
says so in ``counts.gui_ops_included``.

Relationship to the codemap: ``codemap/gen/nodes.jsonl`` and ``sockets.jsonl`` are the
grep-first index (one terse record per line, defaults elided). This file is the READABLE
companion - nested per node, every field spelled out, with a glossary of the types - for a
human reviewing the catalog or a tool that wants one document rather than two indexes.

Run with the venv interpreter (``.venv\\Scripts\\python.exe``): it imports the engine.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT_PATH = os.path.join(ROOT, "codemap", "node_synopsis.json")
ROLES_PATH = os.path.join(ROOT, "codemap", "node_roles.json")
SCHEMA = "nd2studios-node-synopsis/2"


class RolesError(RuntimeError):
    """codemap/node_roles.json disagrees with the registry; the message says how."""


def load_roles(ops: List[str]) -> Dict[str, Any]:
    """The curated role map, validated against the live op list.

    Every shipped op must appear in exactly one role, every op named must exist, and every
    role must name a declared stage. Since V4.00 every role must also list the typed PAGE
    KINDS whose palette offers it (``pages``, all declared in the file's ``pages`` section),
    every ``op_pages`` override must name a real op and real kinds, and every declared kind
    must be used by at least one op. Raising rather than warning is the point: a node added
    without a role would otherwise ship unclassified and nobody would notice."""
    with open(ROLES_PATH, encoding="utf-8") as fh:
        roles = json.load(fh)
    seen: Dict[str, str] = {}
    problems: List[str] = []
    for rname, r in roles["roles"].items():
        if r.get("stage") not in roles["stages"]:
            problems.append(f"role {rname!r} names unknown stage {r.get('stage')!r}")
        for op in r.get("ops", []):
            if op in seen:
                problems.append(f"{op} is in both {seen[op]!r} and {rname!r}")
            seen[op] = rname
    unknown = sorted(set(seen) - set(ops))
    missing = sorted(set(ops) - set(seen))
    if unknown:
        problems.append("ops named in node_roles.json that do not exist: " + ", ".join(unknown))
    if missing:
        problems.append("shipped ops with NO role (add them to node_roles.json): "
                        + ", ".join(missing))
    # V4.00 page kinds
    pages = roles.get("pages")
    if not isinstance(pages, dict) or not pages:
        problems.append("no 'pages' section (V4.00: the typed page kinds, in pipeline order)")
        pages = {}
    used_pages: set = set()
    for rname, r in roles["roles"].items():
        rp = r.get("pages")
        if not isinstance(rp, list) or not rp:
            problems.append(f"role {rname!r} has no 'pages' list (which page kinds offer it?)")
            continue
        bad = [p for p in rp if p not in pages]
        if bad:
            problems.append(f"role {rname!r} names unknown page kind(s) {bad}")
        used_pages.update(rp)
    op_pages = roles.get("op_pages") or {}
    if not isinstance(op_pages, dict):
        problems.append("'op_pages' must be an object {op: [kinds]}")
        op_pages = {}
    for op, ps in op_pages.items():
        if op not in seen:
            problems.append(f"op_pages names {op!r}, which is in no role")
        if not isinstance(ps, list) or not ps:
            problems.append(f"op_pages[{op!r}] must be a non-empty list of page kinds")
            continue
        bad = [p for p in ps if p not in pages]
        if bad:
            problems.append(f"op_pages[{op!r}] names unknown page kind(s) {bad}")
        used_pages.update(ps)
    unused = sorted(set(pages) - used_pages)
    if unused:
        problems.append("page kind(s) no op is offered on: " + ", ".join(unused))
    if problems:
        raise RolesError("codemap/node_roles.json needs attention:\n  " + "\n  ".join(problems))
    return roles

# ── glossary ─────────────────────────────────────────────────────────────────────────────
# Prose a reader needs to interpret the records. Kept here, not in the registry, because the
# registry's enums document the engine's view and this is the reviewer's view.

SOCKET_TYPES: Dict[str, str] = {
    "dataset": "The main payload wire. Carries a `nodegraph.dataset.Dataset`: an image block "
               "over the axes (b, m, t, z, c, y, x) = (batch, multipoint, time, z, channel, "
               "y, x), its calibration metadata (pixel size, z step, frame interval), and an "
               "attribute store of named layers per Domain (voxel rasters, per-label tables, "
               "per-point tables, ...). A node with no dataset input is a source.",
    "float": "A real-valued parameter. May carry a `unit` (um, s, px, ...) and a `derive` "
             "expression that computes a metadata-aware default from the incoming Dataset.",
    "int": "An integer parameter.",
    "bool": "A yes/no parameter.",
    "vector": "A 2- or 3-component numeric parameter (`dims` says which).",
    "color": "An RGB(A) colour parameter.",
    "string": "Free text, OR - when `layer_in`/`layer_out`/`column_in`/`path_kind` are set - "
              "the name of an attribute layer, a structure-table column, or a filesystem "
              "path. The GUI turns those into pickers; the engine sees a string.",
    "menu": "A fixed-choice parameter exposed as a socket (contrast `modes`, which are "
            "in-body dropdowns and not sockets).",
}

SOCKET_FLAGS: Dict[str, str] = {
    "role": "`dataset` = a wire carrying a Dataset; `parameter` = a value socket, typed in "
            "the inspector or driven from an upstream value output.",
    "multi": "The input accepts several wires (payloads arrive in canonical socket order).",
    "is_field": "Blender convention: the value may be wired from an upstream value output "
                "instead of typed. False marks a parameter that must be a literal.",
    "unit": "Physical unit of the value. The engine converts between calibrated units and "
            "pixels using the Dataset's metadata (see concepts CON-05).",
    "derive": "Expression evaluated against the incoming Dataset's metadata to produce the "
              "default (e.g. the 2D/3D lever's `'3D' if (n_z or 1) > 1 else '2D'`).",
    "default": "The literal default when no `derive` applies.",
    "available_in": "{mode: [values]} - the socket is active only while every listed mode "
                    "holds one of the listed values. Hidden sockets keep their value and "
                    "still fold into the memo key.",
    "layer_in": "The STRING names an attribute layer that must already exist on the incoming "
                "Dataset, in this Domain; the GUI offers the layers actually present.",
    "layer_out": "The STRING names a layer this node CREATES, in each listed Domain.",
    "layer_from": "Which Dataset input a `layer_in` socket picks from (empty = the primary).",
    "column_in": "The STRING names a column on the structure table of this Domain.",
    "column_join": "Extra Domains whose columns are also offered, reached by a join.",
    "path_kind": "`open_file` / `save_file` / `directory` - the STRING is a path and the GUI "
                 "shows a Browse button.",
    "view_source": "On an auxiliary Dataset input: its image is composited under the primary "
                   "when this node is viewed.",
    "passes_domains": "On an auxiliary Dataset input: whether that wire's Domains become part "
                      "of what this node's output carries (False = read-only side input).",
}

MODE_FIELDS: Dict[str, str] = {
    "modes": "In-body dropdowns. Not sockets: they reconfigure the node (which sockets are "
             "active, which kernel runs, which footprint applies) and fold into the recipe "
             "hash. `role` = `dim_lever` is the 2D/3D header toggle; `role` = `scope` picks "
             "the statistics population a data-derived parameter is computed over.",
}

FOOTPRINT: Dict[str, str] = {
    "granularity": "How much of the data one kernel call must see, which gates tiling and "
                   "memo granularity: `tileable` (pointwise / small stencil, honours the tile "
                   "provider), `whole_plane` (a full Y,X plane per m,t,z,c), `whole_volume` "
                   "(a full Z,Y,X volume per m,t,c), `whole_series` (the full T series per "
                   "m,c), `multi_view` (several M at once). A mapping means it depends on a "
                   "mode's value.",
    "kernel_axes": "The axes the kernel consumes whole; the complement is what the scheduler "
                   "may split. A mapping means it depends on a mode's value.",
    "footprint_mode": "Which mode's value selects the granularity / kernel_axes entry "
                      "(usually the 2D/3D lever `dim`).",
    "supports_2d": "The node can run plane by plane.",
    "supports_true_3d": "The node has a genuine volumetric path (not a stack of 2D calls).",
    "three_d_fallback": "What the node does in 3D when it has no true-3D path "
                        "(`stack_of_2d` = applies the 2D kernel per plane).",
}

DOMAIN_INTRO = ("The attribute Domains a node reads from the incoming Dataset and the ones it "
                "adds to its output. `reads` is the per-type requirement; `reads_by_mode` "
                "refines it per mode value; `adds` is what appears downstream. Domain meanings "
                "are taken from `nodegraph.domains.Domain`.")


# ── helpers ───────────────────────────────────────────────────────────────────────────────

def _plain(v: Any) -> Any:
    """JSON-native, stably ordered."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if hasattr(v, "value") and hasattr(v, "name") and not isinstance(v, (str, int)):
        return v.value                                   # Enum -> its value
    if isinstance(v, (frozenset, set)):
        return sorted(_plain(x) for x in v)
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if callable(v):
        return getattr(v, "__name__", type(v).__name__)
    return repr(v)


def _squash(text: Any) -> str:
    return " ".join(str(text or "").split())


def _socket(s: Any) -> Dict[str, Any]:
    """Every SocketSpec field, with defaults kept where they carry meaning."""
    rec: Dict[str, Any] = {
        "name": s.name,
        "label": s.label or s.name,
        "type": s.type.value,
        "role": "dataset" if s.type.value == "dataset" else "parameter",
    }
    if s.multi:
        rec["multi"] = True
    if s.type.value != "dataset":
        rec["is_field"] = bool(s.is_field)
        if s.unit:
            rec["unit"] = s.unit
        if s.default is not None:
            rec["default"] = _plain(s.default)
        if s.derive:
            rec["derive"] = s.derive
        if s.type.value == "vector":
            rec["dims"] = s.dims
    for f in dataclasses.fields(s):
        if f.name in ("name", "label", "type", "direction", "multi", "is_field", "unit",
                      "default", "derive", "dims", "description"):
            continue
        v = getattr(s, f.name)
        if v == f.default or v is None or v == () or v == "" or v == {}:
            if not (f.name == "passes_domains" and v is False):
                continue
        rec[f.name] = _plain(v)
    rec["description"] = _squash(s.description)
    return rec


def _mode(m: Any) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "name": m.name,
        "label": m.label or m.name,
        "choices": list(m.choices),
        "default": m.resolved_default(),
    }
    if m.derive:
        rec["derive"] = m.derive
    if m.role:
        rec["role"] = m.role
    if m.presentation != "body":
        rec["presentation"] = m.presentation
    if m.available_in:
        rec["available_in"] = _plain(m.available_in)
    rec["description"] = _squash(m.description)
    if m.choice_docs:
        rec["choice_docs"] = {k: _squash(v) for k, v in m.choice_docs.items()}
    return rec


def _params_read_index():
    """Which param keys each compute actually reads, by compute name - the selftest's own
    resolver, so this file agrees with the gate's definition of 'read'."""
    try:
        from nodegraph.selftest import _param_key_index, _catalog_modules
        return _param_key_index(_catalog_modules())
    except Exception:
        return None


# ── build ─────────────────────────────────────────────────────────────────────────────────

def build() -> Dict[str, Any]:
    gui_ok = True
    try:
        import nodelab_v2.ops as OPS                      # registers io.load / io.dock / view.viewer
        OPS.ensure_ops()
    except Exception:
        gui_ok = False
    import nodegraph.nodes as NN                         # importing registers the catalog
    from nodegraph.registry import NODES
    from nodegraph.hotreload import is_catalog_op
    from nodegraph.domains import Domain
    from nodegraph import codemap as CM

    keys_of = _params_read_index()
    domains = {d.value: _squash(_domain_doc(d)) for d in Domain}

    nodes: Dict[str, Any] = {}
    cats: Dict[str, List[str]] = {}
    n_in = n_out = 0
    for spec in NODES.all():
        owner = NODES.owner(spec.op_key) or ""
        catalog = bool(is_catalog_op(spec.op_key))
        if not (catalog or owner.startswith("nodelab_v2.")):
            continue                                     # test fixtures never ship
        compute = NN.COMPUTES.get(spec.op_key)
        cname = getattr(compute, "__name__", "") if compute is not None else ""
        mod = sys.modules.get(owner)
        mod_path = (getattr(mod, "__file__", "") if mod is not None else "") or ""
        mod_rel = os.path.relpath(mod_path, ROOT).replace(os.sep, "/") if mod_path else ""
        # The long-form prose: the compute's docstring, else the owning module's. A node whose
        # compute is a shared forwarder (the enhance.* family) or that has no compute at all
        # (group/zone ports, the GUI's io.load / view.viewer) documents itself at module level.
        overview = _squash(getattr(compute, "__doc__", "") if compute is not None else "")
        overview_source = "compute"
        if not overview and mod is not None:
            overview = _squash(getattr(mod, "__doc__", ""))
            overview_source = "module"
        if not overview:
            overview_source = "none"

        rec: Dict[str, Any] = {
            "op": spec.op_key,
            "label": spec.label,
            "category": spec.category,
            "module": mod_rel,
            "gui_only": not catalog,
            "summary": _squash(spec.description),
            "overview": overview,
            "overview_source": overview_source,
            "dimensionality": {
                "supports_2d": bool(spec.supports_2d),
                "supports_true_3d": bool(spec.supports_true_3d),
                "three_d_fallback": spec.three_d_fallback or None,
            },
            "footprint": {
                "granularity": _plain(spec.granularity),
                "kernel_axes": _plain(spec.kernel_axes),
                "footprint_mode": spec.footprint_mode,
            },
            "domains": {
                "reads": sorted(d.value for d in spec.reads_domains),
                "reads_by_mode": {
                    mode: {value: sorted(d.value for d in doms)
                           for value, doms in sorted(per.items())}
                    for mode, per in sorted(spec.reads_domains_by_mode.items())},
                "adds": sorted(d.value for d in spec.adds_domains),
            },
            "modes": [_mode(m) for m in spec.modes],
            "inputs": [_socket(s) for s in spec.inputs],
            "outputs": [_socket(s) for s in spec.outputs],
            "compute": cname or None,
            "meta_transform": _plain(spec.meta_transform) if spec.meta_transform else None,
            "extra_layers": _plain(spec.extra_layers) if spec.extra_layers else None,
            "adds_columns": bool(spec.adds_columns),
            "trained_params": bool(spec.trained_params),
            "kernel_doc": CM._kernel_doc(owner) if catalog else None,
            "params_read": sorted(keys_of(cname)) if (keys_of and cname) else [],
        }
        n_in += len(spec.inputs)
        n_out += len(spec.outputs)
        nodes[spec.op_key] = rec
        cats.setdefault(spec.category, []).append(spec.op_key)

    from nodegraph.registry import Granularity

    roles = load_roles(list(nodes))
    role_of = {op: rname for rname, r in roles["roles"].items() for op in r["ops"]}
    for op, rec in nodes.items():
        rname = role_of[op]
        rec["role"] = rname
        rec["stage"] = roles["roles"][rname]["stage"]
        # the typed page kinds whose palette offers this op (V4.00): the per-op override
        # first, else the role's list — the same rule nodegraph.roles.pages_of applies
        rec["pages"] = list((roles.get("op_pages") or {}).get(op)
                            or roles["roles"][rname].get("pages") or [])
        # Put role/stage/pages right after category so a reader sees every grouping together.
        ordered = {}
        for k, v in rec.items():
            ordered[k] = v
            if k == "category":
                ordered["role"] = rec["role"]
                ordered["stage"] = rec["stage"]
                ordered["pages"] = rec["pages"]
        nodes[op] = ordered

    doc = {
        "schema": SCHEMA,
        "generated_by": "scripts/_node_synopsis.py write",
        "source": "nodegraph.registry.NODES after importing nodegraph.nodes"
                  + (" + nodelab_v2.ops.ensure_ops()" if gui_ok else ""),
        "how_to_read": {
            "nodes": "Keyed by op_key. `category` is the GUI palette group; `role` is the "
                     "functional classification and `stage` its pipeline stage, both from "
                     "codemap/node_roles.json (see `roles` and `stages` below). "
                     "`summary` is the node's one-line description; `overview` "
                     "is its compute docstring collapsed to one paragraph (the long-form "
                     "'what and why'), or the owning module's docstring when the compute has "
                     "none (`overview_source` says which). `inputs`/`outputs` are in socket "
                     "order.",
            "socket_types": SOCKET_TYPES,
            "socket_fields": SOCKET_FLAGS,
            "modes": MODE_FIELDS["modes"],
            "footprint": FOOTPRINT,
            "domains": DOMAIN_INTRO,
            "domain_meanings": domains,
            "granularities": sorted(g.value for g in Granularity),
            "params_read": "Param keys the compute actually reads, resolved through shared "
                           "helpers by the selftest's own index. A declared input absent here "
                           "is read only by the engine/GUI (e.g. a layer-name picker).",
            "codemap": "Grep-first index of the same facts: codemap/gen/nodes.jsonl, "
                       "codemap/gen/sockets.jsonl. Concepts: codemap/concepts.md.",
        },
        "counts": {
            "nodes": len(nodes),
            "categories": len(cats),
            "roles": len(roles["roles"]),
            "stages": len(roles["stages"]),
            "inputs": n_in,
            "outputs": n_out,
            "gui_ops_included": gui_ok,
        },
        "stages": {
            s: {**meta, "roles": [r for r, rr in roles["roles"].items() if rr["stage"] == s],
                "ops": sorted(op for r, rr in roles["roles"].items() if rr["stage"] == s
                              for op in rr["ops"])}
            for s, meta in roles["stages"].items()},
        "roles": {
            r: {"label": rr["label"], "stage": rr["stage"], "description": rr["description"],
                "count": len(rr["ops"]),
                "ops": [{"op": op, "label": nodes[op]["label"], "summary": nodes[op]["summary"]}
                        for op in sorted(rr["ops"])]}
            for r, rr in roles["roles"].items()},
        "categories": {c: sorted(ops) for c, ops in sorted(cats.items())},
        "nodes": {k: nodes[k] for k in sorted(nodes)},
    }
    return doc


def _domain_doc(d: Any) -> str:
    """Domain prose from the generated vocabulary (harvested from nodegraph/domains.py)."""
    path = os.path.join(ROOT, "codemap", "gen", "vocab.jsonl")
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("group") == "Domain" and r.get("t") == d.value:
                    return r.get("doc", "")
    except OSError:
        pass
    return ""


def _dump(doc: Dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def main(argv: List[str]) -> int:
    mode = argv[0] if argv else "check"
    if mode == "show":
        if len(argv) < 2:
            print("usage: _node_synopsis.py show <op_key>")
            return 2
        doc = build()
        rec = doc["nodes"].get(argv[1])
        if rec is None:
            print(f"unknown op {argv[1]!r}; known: {', '.join(doc['nodes'])}")
            return 1
        print(_dump(rec))
        return 0

    try:
        doc = build()
    except RolesError as e:
        print(f"SYNOPSIS BLOCKED - {e}")
        return 1
    text = _dump(doc)
    rel = os.path.relpath(OUT_PATH, ROOT).replace(os.sep, "/")
    c = doc["counts"]
    if mode == "write":
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"wrote {rel} - {c['nodes']} nodes, {c['categories']} categories, "
              f"{c['inputs']} inputs, {c['outputs']} outputs"
              + ("" if c["gui_ops_included"] else "  (GUI ops NOT included: PySide6 missing)"))
        return 0
    if mode == "check":
        try:
            with open(OUT_PATH, encoding="utf-8") as fh:
                on_disk = fh.read()
        except OSError:
            print(f"SYNOPSIS MISSING - run: python scripts/_node_synopsis.py write")
            return 1
        if on_disk == text:
            print(f"SYNOPSIS CURRENT - {c['nodes']} nodes, {c['inputs']} inputs, "
                  f"{c['outputs']} outputs")
            return 0
        old = json.loads(on_disk) if on_disk.strip() else {"nodes": {}}
        added = sorted(set(doc["nodes"]) - set(old.get("nodes", {})))
        gone = sorted(set(old.get("nodes", {})) - set(doc["nodes"]))
        changed = sorted(k for k in set(doc["nodes"]) & set(old.get("nodes", {}))
                         if doc["nodes"][k] != old["nodes"][k])
        print(f"SYNOPSIS STALE - {rel} differs from the live registry:")
        for tag, ops in (("added", added), ("removed", gone), ("changed", changed)):
            if ops:
                print(f"  {tag}: {', '.join(ops)}")
        print("  run: python scripts/_node_synopsis.py write")
        return 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
