"""table.curate_objects — the reviewed objects of a table: drop the ones a QC review rejected, merge
the ids a tracker split, by id (2026-10-30)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

import nodegraph.catalog._shared.table_ops as TB
from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _resolve_layer, _structure_layers
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset

_CO_DOMAINS = ("label", "point")
_DEFAULT_ID = "object_uid"
_BEFORE_SUFFIX = "_before"


def _co_ids(text: Any, what: str) -> List[int]:
    """A comma list of integer ids, ``a-b`` for an inclusive range. Raises on anything else."""
    out: List[int] = []
    for tok in ("" if text is None else str(text)).split(","):
        tok = tok.strip()
        if not tok:
            continue
        rng = re.fullmatch(r"(\d+)\s*-\s*(\d+)", tok)
        if rng and int(rng.group(1)) <= int(rng.group(2)):
            out.extend(range(int(rng.group(1)), int(rng.group(2)) + 1))
        elif re.fullmatch(r"\d+", tok):
            out.append(int(tok))
        else:
            raise ValueError(f"Curate Objects: {what} reads {tok!r}, which is not an id or an "
                             f"`a-b` range of ids.")
    return list(dict.fromkeys(out))


def _co_pairs(text: Any) -> Dict[int, int]:
    """``from>to`` pairs, comma separated, resolved through chains (``a>b, b>c`` sends a to c).
    Raises on a malformed pair, an id merged two ways, or a cycle."""
    pairs: Dict[int, int] = {}
    for tok in ("" if text is None else str(text)).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.count(">") != 1:
            raise ValueError(f"Curate Objects: Merge reads {tok!r}; write each merge as "
                             f"`from>to`, e.g. `32>31`.")
        a, b = (s.strip() for s in tok.split(">"))
        if not (re.fullmatch(r"\d+", a) and re.fullmatch(r"\d+", b)):
            raise ValueError(f"Curate Objects: Merge reads {tok!r}, which is not two ids.")
        src, dst = int(a), int(b)
        if src == dst:
            continue
        if src in pairs and pairs[src] != dst:
            raise ValueError(f"Curate Objects: Merge sends {src} to both {pairs[src]} and {dst}.")
        pairs[src] = dst
    resolved: Dict[int, int] = {}
    for src in pairs:
        seen, cur = {src}, pairs[src]
        while cur in pairs:
            if cur in seen:
                raise ValueError(f"Curate Objects: Merge has a cycle through {sorted(seen)}.")
            seen.add(cur)
            cur = pairs[cur]
        resolved[src] = cur
    return resolved


def _compute_curate_objects(ctx: EvalContext) -> Dataset:
    """Drop the objects a review rejected and merge the ids a tracker split, by id.

    Resolved spec (2026-10-30, `build-node-v2`):

    * reads one table (``domain`` label / point, ``table``) and its integer ``id_column``
      (blank: ``object_uid``, which Unique Objects writes — blank rather than that name as the
      socket default, because no structure producer declares it and a closed dropdown would
      show it as an orphan);
    * ``drop`` lists ids (comma list, ``a-b`` ranges) whose rows are removed; it is matched
      against the ORIGINAL ids, before any merge, because a review names the ids it saw;
    * ``merge`` lists ``from>to`` pairs: the rows of ``from`` take id ``to`` (chains resolve,
      a cycle or an id sent two ways is refused). A pair whose target is dropped is refused,
      since the merged rows would carry an id the review rejected;
    * listed ids the table does not carry are skipped and counted in the progress line (a
      subset of passes need not contain every reviewed object);
    * the kept rows keep every column, in the input's row order, plus ``<id_column>_before``,
      the id each row had before the merge;
    * ``name`` blank writes the result back over the table it reads — allowed only for a table
      a table node wrote (Table Concat's, Unique Objects', ...), never a measured table whose
      raster still indexes the rows; otherwise the result is a NEW table ``name``;
    * WHOLE_SERIES, tables only; no axis or calibration change.
    """
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    dom = TB.table_domain(modes)
    layer = _resolve_layer(
        _structure_layers(ds, dom), ctx.layer("table"), node="curate objects",
        socket="table", what=f"{dom.value} table", where="the `data` input",
        remedy=f"wire a node that makes a {dom.value} table (Unique Objects writes "
               f"`object_uid`), or switch Domain", ctx=ctx)[0]
    name = ctx.layer("name") or ""
    if name:
        TB.refuse_voxel_name(ds, name, "Curate Objects")
    elif layer not in TB.synthesized(ds.metadata, dom):
        raise ValueError(
            f"Curate Objects: Name is blank, which writes the result back over {layer!r} — but "
            f"that table was measured, and its raster still indexes the rows a drop would "
            f"remove. Give a Name for a new table.")
    cols = TB.read_table(ds, dom, layer)
    idname = str(ctx.params.get("id_column", _DEFAULT_ID) or _DEFAULT_ID)
    if idname not in cols:
        raise ValueError(f"Curate Objects: the {layer!r} table has no {idname!r} column (it has "
                         f"{sorted(cols)}). Unique Objects writes `object_uid`; pick the id "
                         f"column otherwise.")
    if TB.is_text(cols[idname]):
        raise ValueError(f"Curate Objects: {idname!r} holds text, not ids.")
    raw = np.asarray(cols[idname], dtype=float)
    ok = np.isfinite(raw)
    if np.any(ok & (raw != np.round(raw))):
        raise ValueError(f"Curate Objects: {idname!r} holds fractional values; it must be an "
                         f"integer id column.")
    ids = np.where(ok, raw, -1).astype(np.int64)
    drop = _co_ids(ctx.params.get("drop", ""), "Drop")
    merge = _co_pairs(ctx.params.get("merge", ""))
    bad = sorted(s for s, d in merge.items() if d in set(drop) and s not in set(drop))
    if bad:
        raise ValueError(f"Curate Objects: Merge sends {bad} into an id that Drop removes "
                         f"({[merge[s] for s in bad]}). Drop them too, or merge them elsewhere.")
    present = set(ids[ok].tolist())
    absent = [i for i in list(drop) + list(merge) if i not in present]
    keep = ok & ~np.isin(ids, np.asarray(drop, dtype=np.int64)) if drop else ok.copy()
    keep |= ~ok                                         # a row with no id is never dropped
    new = ids.copy()
    for src, dst in merge.items():
        new[ids == src] = dst
    out = {k: np.asarray(v)[keep] for k, v in cols.items()}
    base = np.asarray(cols[idname])
    out[idname] = np.where(ok, new, base.astype(float) if base.dtype.kind == "f" else new)[keep]
    out[idname + _BEFORE_SUFFIX] = np.asarray(cols[idname])[keep]
    nd = int((~keep).sum())
    nm = int(np.isin(ids[keep], np.asarray(list(merge), dtype=np.int64)).sum()) if merge else 0
    ctx.progress(1, 1, f"{len(ids)} rows → {int(keep.sum())}: dropped {nd} row(s) of "
                       f"{len(set(drop) & present)} id(s), merged {nm} row(s) of "
                       f"{len(set(merge) & present)} id(s)"
                       + (f"; {len(absent)} listed id(s) not in this table" if absent else ""))
    zk = ds.structure_zkind(dom, layer) or "plane_index"
    return TB.write_table(ds, dom, name or layer, out, zk)


def _co_layers(params: Mapping[str, Any], modes: Mapping[str, Any]):
    name = TB.param_text(params, "name", "")
    return ((TB.table_domain(modes), name),) if name else ()


def _co_columns(params: Mapping[str, Any], modes: Mapping[str, Any],
                incoming: Sequence = (), inputs: Sequence = ()):
    """The read table's columns plus ``<id_column>_before``, on the table written."""
    try:
        dom = TB.table_domain(modes)
        table = TB.param_text(params, "table")
        envs = {sock: env for sock, env in (inputs or ())}
        env = envs.get("data")
        if env is not None:
            table = TB.envelope_layer(env, dom, table)
            cols = TB.envelope_columns(env, dom, table)
        else:
            known = list(dict.fromkeys(lyr for d, lyr, _c in incoming or () if d is dom))
            table = table or (known[0] if len(known) == 1 else "")
            cols = [c for d, lyr, c in incoming or () if d is dom and lyr == table]
        name = TB.param_text(params, "name", "") or table
        idc = TB.param_text(params, "id_column", _DEFAULT_ID) or _DEFAULT_ID
        cols = list(cols) + [idc + _BEFORE_SUFFIX]
        return tuple((dom, name, c) for c in dict.fromkeys(cols) if c) if name else ()
    except Exception:                                   # noqa: BLE001 — total by contract
        return ()


register_node(
    _compute_curate_objects, op_key="table.curate_objects", label="Curate Objects",
    category="table",
    inputs=[
        InDataset("data", description="The Dataset whose table is curated; it is passed on "
                                       "with that table rewritten (blank Name) or a new one."),
        InString("table", "Table", field=False, default="", layer_in_mode="domain",
                 description="Which table of the chosen kind holds the objects. Blank uses the "
                             "only one on the wire, and asks when there are several."),
        InString("id_column", "Id column", field=False, default="",
                 column_in_mode="domain", column_from="table",
                 description="The integer column naming each object, the ids Drop and Merge "
                             "refer to. Blank uses `object_uid`, the id Unique Objects writes. "
                             "If anything upstream renumbers it — a new tolerance, another "
                             "input — the lists here point at different objects: review them "
                             "again."),
        InString("drop", "Drop", field=False, default="",
                 description="Ids to remove, comma separated, with `a-b` for a range, e.g. "
                             "`4, 7, 16-18`. Every row of each id leaves the table, so it "
                             "leaves every count, fit and plot downstream. Matched against the "
                             "ids BEFORE Merge. Blank drops nothing."),
        InString("merge", "Merge", field=False, default="",
                 description="Ids that are one object, as `from>to` pairs, comma separated, "
                             "e.g. `32>31`: the rows of 32 take id 31. For a track split when "
                             "its field drifted further than the linking tolerance. Chains "
                             "resolve (`a>b, b>c` sends a to c). Rows keep their measurements; "
                             "only the id changes, and the old one is kept in "
                             "`<id column>_before`. Blank merges nothing."),
        InString("name", "Name", field=False, default="",
                 description="Blank (the default) writes the curated rows back over the table "
                             "it reads, so every node downstream sees them without a change — "
                             "allowed only for a table a table node wrote (Table Concat, Unique "
                             "Objects, ...). A name writes a NEW table of the same kind and "
                             "leaves the input whole."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("domain", list(_CO_DOMAINS), default="label", label="Domain",
             description="Which kind of table holds the objects.",
             choice_docs=domain_docs(_CO_DOMAINS)),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains_by_mode={"domain": {"label": frozenset({Domain.LABEL}),
                                      "point": frozenset({Domain.POINT})}},
    extra_layers=_co_layers, adds_columns=_co_columns,
    description="Apply a review to a table of objects: drop the ids it rejected and merge the "
                "ids that are one object (a track split when its field drifted), keeping each "
                "row's original id beside the new one.")
