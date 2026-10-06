"""table.concat — stack the tables of several inputs into one, each row labelled with its
condition (V4.00 step 9)."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import nodegraph.catalog._shared.table_ops as TB
from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _resolve_layer, _structure_layers
from nodegraph.dataset import Dataset
from nodegraph.domains import Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.metadata import CONDITION_KEY, CONDITION_SET_KEY
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset

_CONCAT_DOMAINS = ("label", "point")
_DEFAULT_NAME = "combined"


def _compute_table_concat(ctx: EvalContext) -> Dataset:
    """One table out of the same-kind tables of every input, rows labelled with their source.

    Resolved spec (V4.00 step 9, `build-node-v2`):

    * ``data`` takes any number of Datasets (the order they were wired in); ``domain`` label |
      point picks which kind of table; ``table`` names it on EVERY input (blank: each input's
      only one of that kind);
    * the output is input 0's Dataset plus the new ``name`` table (default ``combined``):
      the union of the inputs' columns — a column an input lacks is NaN (numbers) or ``""``
      (text) for its rows — plus ``condition`` (the input's ``condition`` metadata, which a
      Page Output stamps; else that input's entry in ``names``; else ``input_K``), ``input``
      (K, the wiring position) and, from each input's OWN per-position metadata, ``group``,
      ``position_name`` and ``file`` when any input carries them;
    * ids stay unique: input K's ids are shifted by the largest id of the inputs before it
      (input 0's ids are unchanged, so it still joins against its own raster and tables);
    * text columns are numpy unicode (memo- and dock-safe); the 2D/3D kind (``z_kind``) must
      agree across inputs; a batch is combined only on its own (one input);
    * no image work: WHOLE_SERIES, nothing read but the tables; no axis or calibration change.
    """
    files: Tuple[Dataset, ...] = tuple(ctx.input("data") or ())
    if not files:
        raise ValueError("Table Concat: wire at least one Dataset into Inputs.")
    modes = ctx.params.get("__modes__", {}) or {}
    domain = TB.table_domain(modes)
    want = ctx.layer("table")
    name = ctx.layer("name") or _DEFAULT_NAME
    raw_names = str(ctx.params.get("names", "") or "").strip()
    names = [s.strip() for s in raw_names.split(",")] if raw_names else []
    if len(files) > 1 and any(int(getattr(f.axes, "b", 1) or 1) > 1 for f in files):
        raise ValueError(
            "Table Concat: one of the inputs is a batch of several files. A batch's rows are "
            "addressed by their batch member, which another input's rows have no part in — "
            "combine a batch with this node on its own, or split it first.")
    TB.refuse_voxel_name(files[0], name, "Table Concat")
    tables: List[Dict[str, np.ndarray]] = []
    zkinds = set()
    for k, ds in enumerate(files):
        layer = _resolve_layer(
            _structure_layers(ds, domain), want, node="table concat", socket="table",
            what=f"{domain.value} table", where=f"input {k}",
            remedy=f"wire a node that makes a {domain.value} table into input {k}, or "
                   f"switch Domain", ctx=ctx)[0]
        cols = TB.read_table(ds, domain, layer)
        if "id" not in cols:
            raise ValueError(f"Table Concat: the {layer!r} table of input {k} has no `id` "
                             f"column — it is not a {domain.value} table this node can "
                             f"combine.")
        tables.append(cols)
        zkinds.add(ds.structure_zkind(domain, layer) or "")
    zkinds.discard("")
    if len(zkinds) > 1:
        raise ValueError(
            "Table Concat: the inputs mix 2D (per-plane) and 3D tables; their `z` columns "
            "mean different things, so one table of both would be wrong in z.")
    # ── ids: input K shifted PAST every id already emitted (0-based Point ids too) ──
    next_free = None
    ids: List[np.ndarray] = []
    for k, cols in enumerate(tables):
        raw = np.asarray(cols["id"])
        i64 = np.asarray(np.where(np.isfinite(raw.astype(float)), raw, 0), dtype=np.int64)
        if k > 0 and i64.size and next_free is not None:
            i64 = i64 + max(0, next_free - int(i64.min()))
        ids.append(i64)
        if i64.size:
            top = int(i64.max()) + 1
            next_free = top if next_free is None else max(next_free, top)
    # ── the columns: input 0's order, then any new ones, missing filled ─────────────
    order: List[str] = []
    for cols in tables:
        order += [c for c in cols if c not in order]
    sizes = [len(np.asarray(cols["id"])) for cols in tables]
    out: Dict[str, np.ndarray] = {}
    for col in order:
        if col == "id":
            out["id"] = np.concatenate(ids) if ids else np.zeros(0, np.int64)
            continue
        like = next(np.asarray(c[col]) for c in tables if col in c)
        parts = [np.asarray(c[col]) if col in c else TB.missing_like(like, n)
                 for c, n in zip(tables, sizes)]
        out[col] = TB.stack_columns(parts, col, "Table Concat")
    # ── provenance ───────────────────────────────────────────────────────────────
    conds: List[str] = []
    for k, ds in enumerate(files):
        c = str((ds.metadata or {}).get(CONDITION_KEY) or "").strip()
        if not c and k < len(names) and names[k]:
            c = names[k]
        conds.append(c or f"input_{k}")

    def per_input(col: str, derive) -> np.ndarray:
        # a row that already SAYS where it came from (a table concatenated before) keeps it
        return TB.text_column(
            v for k, (cols, n) in enumerate(zip(tables, sizes))
            for v in ([str(x) for x in np.asarray(cols[col]).tolist()] if col in cols
                      else derive(k, cols, n)))

    out[TB.CONDITION_COLUMN] = per_input(TB.CONDITION_COLUMN,
                                         lambda k, cols, n: [conds[k]] * n)
    out[TB.INPUT_COLUMN] = np.concatenate(
        [np.full(n, k, dtype=np.int64) for k, n in enumerate(sizes)]) if sizes \
        else np.zeros(0, np.int64)
    batched = [int(getattr(ds.axes, "b", 1) or 1) > 1 for ds in files]
    for key, col in TB.PER_M_COLUMNS:
        # a batch's per-M lists describe ONE member: its rows are named per member by the
        # tabulator (by `b`), so the list is not read here
        got = [None if batched[k] else
               TB.per_m_text(ds.metadata or {}, key, np.asarray(cols.get("m", np.zeros(n))))
               for k, (ds, cols, n) in enumerate(zip(files, tables, sizes))]
        if all(g is None for g in got) and not any(col in cols for cols in tables):
            continue
        out[col] = per_input(col, lambda k, cols, n, got=got:
                             got[k].tolist() if got[k] is not None else [""] * n)
    # each row's own clock, when EVERY input has one (a row with no clock beside rows with
    # one could not be placed on a shared time axis honestly)
    clocks = [TB.frame_seconds(ds.metadata or {}, np.asarray(cols.get("t", np.zeros(n))))
              for ds, cols, n in zip(files, tables, sizes)]
    if all(c is not None for c in clocks) and not any(TB.TIME_COLUMN in c for c in tables):
        out[TB.TIME_COLUMN] = np.concatenate(clocks) if clocks else np.zeros(0)
        out[TB.CLOCK_COLUMN] = np.concatenate(
            [TB.frame_jd(ds.metadata or {}, np.asarray(cols.get("t", np.zeros(n))))
             for ds, cols, n in zip(files, tables, sizes)]) if clocks else np.zeros(0)
    zk = next(iter(zkinds), "plane_index")
    ctx.progress(1, 1, f"{sum(sizes)} rows from {len(files)} input(s): "
                       + ", ".join(f"{c} ({n})" for c, n in zip(conds, sizes)))
    # the output holds several conditions: input 0's scalar `condition` no longer describes
    # it (a blank Page Output downstream would otherwise keep it as if typed)
    base = files[0].with_metadata(**{CONDITION_KEY: None, CONDITION_SET_KEY: None})
    return TB.write_table(base, domain, name, out, zk)


def _concat_meta(env, params: Mapping[str, Any], modes: Mapping[str, Any]):
    """Edit-time twin of the compute's metadata change: the scalar condition is cleared."""
    return env.with_metadata(**{CONDITION_KEY: None, CONDITION_SET_KEY: None})


def _concat_layers(params: Mapping[str, Any], modes: Mapping[str, Any]):
    """The table this node creates — its domain follows the ``domain`` Mode (total)."""
    return ((TB.table_domain(modes), TB.param_text(params, "name", _DEFAULT_NAME)),)


def _concat_columns(params: Mapping[str, Any], modes: Mapping[str, Any],
                    incoming: Sequence = (), inputs: Sequence = ()):
    """The columns of the ``name`` table: every input's chosen table's columns, ``condition``
    and ``input``, and the per-position columns any input's metadata carries (total)."""
    try:
        dom = TB.table_domain(modes)
        name = TB.param_text(params, "name", _DEFAULT_NAME)
        want = TB.param_text(params, "table")
        envs = [env for _sock, env in (inputs or ())]
        cols: List[str] = []
        for env in envs:
            cols += TB.envelope_columns(env, dom, TB.envelope_layer(env, dom, want))
        if not envs:
            known = list(dict.fromkeys(lyr for d, lyr, _c in incoming or () if d is dom))
            layer = want or (known[0] if len(known) == 1 else "")
            cols += [c for d, lyr, c in incoming or () if d is dom and lyr == layer]
        cols += [TB.CONDITION_COLUMN, TB.INPUT_COLUMN]
        if envs and all(TB.envelope_has(env, "frame_time_jd") for env in envs):
            cols += [TB.TIME_COLUMN, TB.CLOCK_COLUMN]
        cols += [col for key, col in TB.PER_M_COLUMNS
                 if any(TB.envelope_has(env, key) and
                        int(getattr(getattr(env, "axes", None), "b", 1) or 1) <= 1
                        for env in envs)]
        return tuple((dom, name, c) for c in dict.fromkeys(cols) if c)
    except Exception:                                   # noqa: BLE001 — total by contract
        return ()


_concat_columns.wants_inputs = True


register_node(
    _compute_table_concat, op_key="table.concat", label="Table Concat", category="table",
    inputs=[
        InDataset(multi=True, label="Inputs", passes_domains=False,
                  description="The Datasets whose tables are stacked — typically one Page "
                              "Input per page or variant. Order is the order you wired them "
                              "in (the first is input 0, whose Dataset is passed on)."),
        InString("table", "Table", field=False, default="", layer_in_mode="domain",
                 description="Which table of the chosen kind to take from EVERY input. Blank "
                             "takes each input's only one, and asks when an input has "
                             "several."),
        InString("names", "Condition names", field=False, default="",
                 description="Comma-separated condition labels, one per input in wiring "
                             "order, used only for an input that carries no `condition` of "
                             "its own (a Page Output stamps one). Blank entries fall back to "
                             "input_0, input_1, ... Changes the labels, never a measurement."),
        InString("name", "Name", field=False, default=_DEFAULT_NAME,
                 description="Name of the combined table this node writes on the output. The "
                             "inputs' own tables are left as they are; reusing an existing "
                             "table's name replaces that table on the output."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("domain", list(_CONCAT_DOMAINS), default="label", label="Domain",
             description="Which kind of table is stacked: one row per labelled region, or "
                         "one per detected point.",
             choice_docs=domain_docs(_CONCAT_DOMAINS)),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains_by_mode={"domain": {"label": frozenset({Domain.LABEL}),
                                      "point": frozenset({Domain.POINT})}},
    extra_layers=_concat_layers, adds_columns=_concat_columns, meta_transform=_concat_meta,
    description="Stack the Label or Point tables of several inputs (pages, conditions, "
                "variants) into one table, each row labelled with its condition, its input "
                "and its position group, ids kept unique — the table a plot groups by "
                "condition.")
