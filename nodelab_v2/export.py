"""Structure-table export (Phase 6 / V2.00 §10 — "the export path reads from here").

Takes the per-``(domain, layer)`` tables that :func:`nodelab_v2.tables.all_tables`
produces off a pulled Dataset and writes them to disk — the detected structures AND (since
V2.17) the coarse lattice attributes, so anything the spreadsheet can show is also
exportable. Reading through the same one function is what keeps those two in step.
Genuinely Qt-free (a caller supplies the path): the tabulation is imported from
:mod:`nodelab_v2.tables`, not from the Qt spreadsheet panel that used to host it — which
is what lets the headless LabLink worker write a session's CSV without a display. numpy +
stdlib for CSV, pyarrow **lazily imported** for Arrow/Parquet so the core has no hard
pyarrow dependency.

* ``.csv`` — one file; when a Dataset has several structure tables the domain/layer is
  written as leading ``domain``/``layer`` columns so everything lands in one sheet.
* ``.parquet`` / ``.arrow`` — a single table (same long form), via pyarrow.

Column order follows :func:`nodelab_v2.spreadsheet` (coordinate columns first).

**Every export says where it came from (2026-10-22).** :func:`export_dataset` writes an
ORIGIN RECORD (:func:`nodegraph.table_file.origin_record`): when and on which machine, the
graph file, page and node, the chain of nodes in front of it (:func:`export_context`), the
source image file(s), the axes, the metadata and a placement per table — inside a Parquet /
Arrow file's schema metadata, beside a CSV as ``<file>.origin.json``. It is what lets Load
Table (``io.load_table``) put every table back on its original layer and offer to load the
original image under it.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from nodegraph import table_file as TF
from nodelab_v2.tables import (
    Tables, _ordered_columns, all_tables, lattice_tables, structure_tables,
)


def _leading_keys(tables: Tables) -> Tuple[str, str]:
    """The long form's two leading column names: ``domain`` and ``layer``, prefixed with
    ``_`` until neither collides with an attribute column (real tables never do, but the
    origin record names whatever was used, so a reader never has to guess)."""
    names = {n for cols in tables.values() for n in cols}
    dom_key, lay_key = "domain", "layer"
    while dom_key in names:
        dom_key = "_" + dom_key
    while lay_key in names:
        lay_key = "_" + lay_key
    return dom_key, lay_key


def _long_rows(tables: Tables) -> Tuple[List[str], List[list]]:
    """Flatten all tables into (header, rows) long form: leading ``domain``/``layer``
    columns then the union of attribute columns (coordinate columns first)."""
    attr_names: List[str] = []
    for cols in tables.values():
        for n in _ordered_columns(cols):
            if n not in attr_names:
                attr_names.append(n)
    # keep coordinate columns first across the union
    attr_names = _ordered_columns({n: None for n in attr_names})
    # the injected leading columns must not collide with an attribute column literally
    # named "domain"/"layer" (real structure tables never do, but stay robust): prefix
    # with "_" until unique.
    dom_key, lay_key = _leading_keys(tables)
    header = [dom_key, lay_key, *attr_names]
    rows: List[list] = []
    for (dom, layer), cols in sorted(tables.items()):
        n = max((len(v) for v in cols.values()), default=0)
        for r in range(n):
            row = [dom, layer if layer is not None else ""]
            for name in attr_names:
                # None (not "") for a missing cell: csv renders it empty, and Arrow
                # keeps a numeric column numeric (a "" would force str and fail typing)
                v = cols[name][r] if name in cols and r < len(cols[name]) else None
                row.append(v.item() if isinstance(v, np.generic) else v)
            rows.append(row)
    return header, rows


def export_csv(tables: Tables, path: str,
               origin: Optional[Mapping[str, Any]] = None) -> int:
    """Write the tables as one long-form CSV; returns the row count. With an ``origin``
    record it is written beside the file as ``<path>.origin.json`` (a CSV has no metadata
    slot); without one, a stale sidecar from an earlier export to that name is removed —
    it would describe a file that no longer exists."""
    header, rows = _long_rows(tables)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    side = TF.sidecar_path(path)
    if origin is not None:
        with open(side, "w", encoding="utf-8") as f:
            json.dump(origin, f, ensure_ascii=False, indent=1)
    elif os.path.isfile(side):
        os.remove(side)
    return len(rows)


def export_arrow(tables: Tables, path: str,
                 origin: Optional[Mapping[str, Any]] = None) -> int:
    """Write the tables as a single Parquet/Arrow file (pyarrow lazily imported), the
    ``origin`` record (when given) in its schema metadata under
    :data:`~nodegraph.table_file.ORIGIN_META_KEY`."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    header, rows = _long_rows(tables)
    columns = {name: [row[i] for row in rows] for i, name in enumerate(header)}
    table = pa.table(columns)
    if origin is not None:
        table = table.replace_schema_metadata(
            {**(table.schema.metadata or {}), TF.ORIGIN_META_KEY: TF.encode_origin(origin)})
    if path.endswith(".arrow"):
        with pa.OSFile(path, "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
    else:
        pq.write_table(table, path)
    return len(rows)


def derived_columns(dataset, tables: Tables) -> Dict[Tuple[str, Optional[str]], List[str]]:
    """Per table, the columns :func:`~nodelab_v2.tables.all_tables` ADDED from metadata
    (``file``, ``group``) rather than read off a stored attribute — what the origin record
    tells a reload to re-derive instead of storing twice."""
    raw = structure_tables(dataset)
    raw.update(lattice_tables(dataset))
    return {key: [c for c in cols if c not in raw.get(key, {})]
            for key, cols in tables.items()}


def origin_for(dataset, tables: Tables,
               context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The origin record for exporting ``tables`` off ``dataset`` (see the module doc)."""
    dom_key, lay_key = _leading_keys(tables)
    return TF.origin_record(dataset, tables, derived=derived_columns(dataset, tables),
                            context=context, columns={"domain": dom_key, "layer": lay_key})


def export_dataset(dataset, path: str, context: Optional[Mapping[str, Any]] = None) -> int:
    """Export a pulled Dataset's tabulatable attributes to ``path`` (extension picks the
    format) — structure tables and coarse lattice attributes alike — with its origin record
    (``context``: what only the caller knows, see :func:`export_context`). Raises
    :class:`ValueError` if the Dataset carries neither kind of table."""
    tables = all_tables(dataset)
    if not tables:
        raise ValueError(
            "nothing tabulatable on this output to export — it carries no structure "
            "tables (Label / Point / Track / Mesh) and no coarse lattice attributes "
            "(Frame / Plane / Timepoint / Multipoint / Channel / Global). Per-voxel image "
            "layers are not exported here; view them in the Viewer, or move one to a "
            "coarser domain with transform.transfer_domain first.")
    origin = origin_for(dataset, tables, context)
    lower = path.lower()
    if lower.endswith((".parquet", ".arrow")):
        return export_arrow(tables, path, origin)
    return export_csv(tables, path, origin)


# ── what only the GUI knows: graph, page, node, chain ───────────────────────────────────

#: A chain entry's param larger than this (as JSON) is summarised, not written — a drawn
#: region list or a channel table has no business in a header.
_PARAM_LIMIT = 4096


def _app_info() -> Dict[str, str]:
    """``{"name", "commit"}`` — the short git commit of this checkout when git can say it
    (a second at most; nothing when it cannot)."""
    info = {"name": "ND2Studios_Blender"}
    try:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        got = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                             capture_output=True, text=True, timeout=2)
        if got.returncode == 0 and got.stdout.strip():
            info["commit"] = got.stdout.strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort
        pass
    return info


def _chain_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (params or {}).items():
        if str(k).startswith("__"):
            continue
        try:
            safe = TF.json_safe(v)
            text = json.dumps(safe)
        except Exception:  # noqa: BLE001 — a value with no JSON form is summarised
            out[str(k)] = f"<{type(v).__name__}>"
            continue
        out[str(k)] = safe if len(text) <= _PARAM_LIMIT else f"<{len(text)} chars>"
    return out


def export_context(workspace, page_id: Optional[str], node_id: str,
                   graph_path: Optional[str] = None) -> Dict[str, Any]:
    """What the origin record needs from the GUI: the app and commit, the graph file, the
    page (id, name, kind), the node (id, op, title) and the CHAIN in front of it — every
    ancestor in the composed run graph (so across pages), sources first, each as
    ``{"node", "op", "label", "params"}``. Qt-free and total: a page that will not compose
    (an unbound Page Input) gives no chain rather than no export."""
    from nodelab_v2.document import TITLE_KEY
    ctx: Dict[str, Any] = {"app": _app_info(), "graph": os.path.abspath(graph_path)
                           if graph_path else ""}
    pages = getattr(workspace, "pages", {}) or {}
    pid = page_id or getattr(workspace, "active", None) or ""
    page = pages.get(pid)
    if page is None:
        ctx["node"] = {"id": node_id}
        return ctx
    ctx["page"] = {"id": pid, "name": str(getattr(page, "name", "") or ""),
                   "kind": str(getattr(page, "kind", "") or "")}
    doc = page.doc
    rec = doc.nodes.get(node_id)
    if rec is not None:
        spec = rec.spec()
        ctx["node"] = {"id": node_id, "op": rec.op_key,
                       "label": str(rec.params.get(TITLE_KEY) or
                                    (spec.label if spec is not None else rec.op_key))}
    else:
        ctx["node"] = {"id": node_id}
    try:
        cg = workspace.compose(pid)
        run = cg.id_map.get((pid, node_id))
        graph = cg.graph
        if run is None or run not in graph.nodes:
            return ctx
        order: List[str] = []
        seen: set = set()

        def visit(nid: str) -> None:
            if nid in seen:
                return
            seen.add(nid)
            for e in graph.preds(nid):
                visit(e.src)
            order.append(nid)

        visit(run)
        chain = []
        for nid in order[:-1]:
            node = graph.nodes[nid]
            spec = node.spec()
            label = str(node.params.get(TITLE_KEY) or
                        (spec.label if spec is not None else node.op_key))
            chain.append({"node": nid, "op": node.op_key, "label": label,
                          "params": _chain_params(node.params)})
        ctx["chain"] = chain
    except Exception:  # noqa: BLE001 — provenance is best-effort, never a failed export
        pass
    return ctx


__all__ = ["export_dataset", "export_csv", "export_arrow", "structure_tables",
           "all_tables", "derived_columns", "origin_for", "export_context"]
