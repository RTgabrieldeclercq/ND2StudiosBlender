"""table.aggregate — one row per group: a column summarised by mean, sem, count, ...
(V4.00 step 9)."""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import nodegraph.catalog._shared.table_ops as TB
from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _resolve_layer, _structure_layers
from nodegraph.dataset import Dataset
from nodegraph.domains import BATCH_COLUMN, Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.metadata import POSITION_GROUP_KEY, SOURCE_FILE_KEY
from nodegraph.reducers import reduce as _reduce, reducer_docs
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset

_AGG_DOMAINS = ("label", "point", "track")
_AGG_REDUCERS = ("mean", "median", "sum", "min", "max", "count", "std", "sem")
_AGG_DEFAULT_REDUCERS = "mean,sem,count"
_DEFAULT_NAME = "summary"
#: the coordinate columns every output row carries (the Label table invariant)
_AGG_COORDS = ("m", "t", "c", "z", "y", "x")
_AGG_INT_COORDS = ("m", "t", "c")
_MISSING = object()


def _agg_list(text: Any) -> List[str]:
    """A comma list as its distinct non-blank entries, in order (total)."""
    try:
        raw = "" if text is None else str(text)
    except Exception:                                   # noqa: BLE001 — total by contract
        return []
    return list(dict.fromkeys(s.strip() for s in raw.split(",") if s.strip()))


def _agg_reducers(text: Any) -> List[str]:
    """The reducer list; blank → the default (mean, sem, count)."""
    names = _agg_list(text) or _agg_list(_AGG_DEFAULT_REDUCERS)
    bad = [n for n in names if n not in _AGG_REDUCERS]
    if bad:
        raise ValueError(f"Table Aggregate: unknown reducer(s) {bad} — choose from "
                         f"{list(_AGG_REDUCERS)}.")
    return names


def _agg_key(v: Any):
    """A group-key cell as a hashable value; one sentinel for every missing one."""
    if v is None:
        return _MISSING
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            return _MISSING
        return int(f) if f.is_integer() else f
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (str, np.str_)):
        return str(v) if str(v).strip() else _MISSING
    return v


def _agg_order(key: Tuple) -> Tuple:
    def one(v):
        if v is _MISSING:
            return (2, 0.0, "")
        if isinstance(v, (int, float)):
            return (0, float(v), "")
        return (1, 0.0, str(v))
    return tuple(one(v) for v in key)


def _compute_table_aggregate(ctx: EvalContext) -> Dataset:
    """A summary table: one row per distinct ``group_by`` value, a column reduced over it.

    Resolved spec (V4.00 step 9, `build-node-v2`):

    * reads one table (``domain``, ``table``); ``group_by`` is a comma list of its columns
      (``t`` for one row per frame, ``condition,t`` per condition and frame; blank = ONE row
      for the whole table); a missing key value (NaN, blank text) is a group of its own,
      sorted last; the batch member ``b`` is always part of the key when the table has it;
    * ``value`` (numeric) is reduced per group by each of ``reducers`` — mean, median, sum,
      min, max, count (finite values), std (n - 1, NaN under two), sem (std / sqrt n) — into
      ``<value>_<reducer>``; ``n`` counts each group's rows;
    * the output is a LABEL table ``name`` (default ``summary``) — the domain every plot and
      export reads — keeping the invariant ``id, m, t, c, z, y, x``: ``id`` 1..K; a
      coordinate in ``group_by`` carries the key (``-1`` where it is missing), any other is
      0, because a summary row has no position of its own; other key columns are carried
      (text as numpy unicode); ``z_kind`` plane_index;
    * WHOLE_SERIES, tables only; no axis or calibration change.
    """
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    dom = TB.table_domain(modes)
    name = ctx.layer("name") or _DEFAULT_NAME
    TB.refuse_voxel_name(ds, name, "Table Aggregate")
    layer = _resolve_layer(
        _structure_layers(ds, dom), ctx.layer("table"), node="table aggregate",
        socket="table", what=f"{dom.value} table", where="the `data` input",
        remedy=f"wire a node that makes a {dom.value} table, or switch Domain", ctx=ctx)[0]
    if dom is Domain.LABEL and name == layer:
        raise ValueError(
            f"Table Aggregate: Name {name!r} is the table it summarises — the summary would "
            f"replace the rows it came from. Choose another Name (the default is "
            f"{_DEFAULT_NAME!r}).")
    cols = TB.read_table(ds, dom, layer)
    keys = _agg_list(ctx.params.get("group_by", "t"))
    unknown = [k for k in keys if k not in cols]
    if unknown:
        raise ValueError(f"Table Aggregate: Group by names {unknown}, which the {layer!r} "
                         f"table does not carry — it has {sorted(cols)}.")
    if BATCH_COLUMN in cols and BATCH_COLUMN not in keys:
        keys = [BATCH_COLUMN] + keys
    vname = str(ctx.params.get("value", "area") or "area")
    if vname not in cols:
        raise ValueError(f"Table Aggregate: the {layer!r} table has no column {vname!r} for "
                         f"Value — it carries {sorted(cols)}.")
    if TB.is_text(cols[vname]):
        raise ValueError(f"Table Aggregate: {vname!r} holds text, not numbers — pick a "
                         f"numeric Value (a text column belongs in Group by).")
    reducers = _agg_reducers(ctx.params.get("reducers", _AGG_DEFAULT_REDUCERS))
    n = len(np.asarray(cols[vname]))
    vals = np.asarray(cols[vname], dtype=float)
    vals = np.where(np.isfinite(vals), vals, np.nan)   # every reducer counts finite values
    kcols = [np.asarray(cols[k]).tolist() for k in keys]
    groups: Dict[Tuple, List[int]] = {}
    for i in range(n):
        groups.setdefault(tuple(_agg_key(c[i]) for c in kcols), []).append(i)
    order = sorted(groups, key=_agg_order)
    k_out = len(order)
    out: Dict[str, np.ndarray] = {"id": np.arange(1, k_out + 1, dtype=np.int64)}
    for c in _AGG_COORDS:
        if c in keys:
            j = keys.index(c)
            raw = [key[j] for key in order]
            if c in _AGG_INT_COORDS:
                out[c] = np.array([-1 if v is _MISSING else int(v) for v in raw],
                                  dtype=np.int64)
            else:
                out[c] = np.array([-1.0 if v is _MISSING else float(v) for v in raw])
        else:
            out[c] = (np.zeros(k_out, dtype=np.int64) if c in _AGG_INT_COORDS
                      else np.zeros(k_out))
    for j, k in enumerate(keys):
        if k in _AGG_COORDS:
            continue
        raw = [key[j] for key in order]
        if k == BATCH_COLUMN:
            out[k] = np.array([0 if v is _MISSING else int(v) for v in raw], dtype=np.int64)
        elif TB.is_text(cols[k]):
            out[k] = TB.text_column("" if v is _MISSING else v for v in raw)
        else:
            out[k] = np.array([np.nan if v is _MISSING else float(v) for v in raw])
    # a summary row pools positions: it carries the ONE file / group its rows share, else
    # blank — never position 0's, which the tabulator would fill in (m = 0 on every row)
    if "m" not in keys:
        batched = BATCH_COLUMN in keys

        def pooled(col: str, key: str) -> np.ndarray:
            if col in cols:
                src = [str(v) for v in np.asarray(cols[col]).tolist()]
            elif "m" in cols and not batched:
                got = TB.per_m_text(ds.metadata or {}, key, np.asarray(cols["m"]))
                src = got.tolist() if got is not None else None
            else:
                src = None
            vals = []
            for gk in order:
                seen = {src[i] for i in groups[gk] if src[i]} if src is not None else set()
                vals.append(next(iter(seen)) if len(seen) == 1 else "")
            return TB.text_column(vals)

        if "group" not in keys:
            out["group"] = pooled("group", POSITION_GROUP_KEY)
        if "file" not in keys and not batched:
            out["file"] = pooled("file", SOURCE_FILE_KEY)
    out["n"] = np.array([len(groups[key]) for key in order], dtype=np.int64)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for r in reducers:
            got = [_reduce(vals[groups[key]], (0,), r) for key in order]
            arr = np.array([float(np.asarray(g)) for g in got]) if got else np.zeros(0)
            out[f"{vname}_{r}"] = arr.astype(np.int64) if r == "count" else arr
    ctx.progress(1, 1, f"{n} rows → {k_out} group(s) by {', '.join(keys) or 'nothing'}")
    return TB.write_table(ds, Domain.LABEL, name, out, "plane_index")


def _agg_layers(params: Mapping[str, Any], modes: Mapping[str, Any]):
    return ((Domain.LABEL, TB.param_text(params, "name", _DEFAULT_NAME)),)


def _agg_columns(params: Mapping[str, Any], modes: Mapping[str, Any], incoming: Sequence = ()):
    """id + the coordinates, the non-coordinate group keys, ``n`` and one column per reducer
    (total; a reducer name not on the menu is skipped, the compute refuses it)."""
    try:
        name = TB.param_text(params, "name", _DEFAULT_NAME)
        p = params or {}
        keys = _agg_list(p.get("group_by", "t") if "group_by" in p else "t")
        value = TB.param_text(params, "value", "area")
        reds = [r for r in (_agg_list(p.get("reducers")) or
                            _agg_list(_AGG_DEFAULT_REDUCERS)) if r in _AGG_REDUCERS]
        cols = ["id", *_AGG_COORDS] + [k for k in keys if k not in _AGG_COORDS]
        if "m" not in keys and "group" not in keys:
            cols.append("group")       # blank on a pooled row (`file` too, but not on a batch)
        cols.append("n")
        cols += [f"{value}_{r}" for r in reds]
        return tuple((Domain.LABEL, name, c) for c in dict.fromkeys(cols) if c)
    except Exception:                                   # noqa: BLE001 — total by contract
        return ()


register_node(
    _compute_table_aggregate, op_key="table.aggregate", label="Table Aggregate",
    category="table",
    inputs=[
        InDataset("data", description="The Dataset whose table is summarised; it is passed "
                                       "on with the summary table added."),
        InString("table", "Table", field=False, default="", layer_in_mode="domain",
                 description="Which table of the chosen kind to summarise. Blank uses the "
                             "only one on the wire, and asks when there are several."),
        InString("group_by", "Group by", field=False, default="t",
                 description="Comma-separated columns whose distinct value combinations "
                             "become the rows: `t` gives one row per frame, `condition,t` one "
                             "per condition and frame, blank ONE row for the whole table. "
                             "More columns give more, smaller groups (and noisier means)."),
        InString("value", "Value", field=False, default="area", column_in_mode="domain",
                 column_from="table",
                 description="The numeric column summarised per group. The summary reads it "
                             "as it is; it changes no measurement upstream."),
        InString("reducers", "Reducers", field=False, default=_AGG_DEFAULT_REDUCERS,
                 vocab=_AGG_REDUCERS, choice_docs=reducer_docs(_AGG_REDUCERS),
                 description="The summaries computed per group, each its own column "
                             "`<value>_<reducer>`. mean with sem is the usual "
                             "condition-comparison pair; std describes the spread itself."),
        InString("name", "Name", field=False, default=_DEFAULT_NAME,
                 description="Name of the summary table — a Label table, so every plot and "
                             "export reads it. The input table is left as it is."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("domain", list(_AGG_DOMAINS), default="label", label="Domain",
             description="Which kind of table is summarised.",
             choice_docs=domain_docs(_AGG_DOMAINS)),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains_by_mode={"domain": {"label": frozenset({Domain.LABEL}),
                                      "point": frozenset({Domain.POINT}),
                                      "track": frozenset({Domain.TRACK})}},
    adds_domains=frozenset({Domain.LABEL}),
    extra_layers=_agg_layers, adds_columns=_agg_columns,
    description="Summarise a Label, Point or Track table per group — one row per frame, "
                "condition or position — with mean, median, sum, min, max, count, std or sem "
                "of a column, as a new Label table a plot can read.")
