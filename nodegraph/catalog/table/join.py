"""table.join — add another table's columns to a table, row by row on a shared key
(V4.00 step 9)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import nodegraph.catalog._shared.table_ops as TB
from nodegraph.catalog._base import register_node
from nodegraph.catalog._shared.labels import _resolve_layer, _structure_layers
from nodegraph.dataset import Dataset
from nodegraph.domains import BATCH_COLUMN, Domain, domain_docs
from nodegraph.engine import EvalContext
from nodegraph.registry import Granularity, InDataset, InString, Mode, OutDataset
from nodegraph.structure import COORD_COLUMNS

_JOIN_DOMAINS = ("label", "point", "track")
#: what each ``on`` choice matches rows by
_JOIN_KEYSETS: Dict[str, Tuple[str, ...]] = {
    "id_m_t": ("id", "m", "t"),
    "id": ("id",),
    "m_t": ("m", "t"),
    "track": ("track_id", "t"),
}
_DEFAULT_NAME = "joined"
_DEFAULT_PREFIX = "other_"
#: never copied from the other table: its coordinates and batch member describe the SAME
#: object (or frame) the key already matched, so they would only repeat the left's
_JOIN_SKIP = frozenset(COORD_COLUMNS) | {BATCH_COLUMN}


def _key_value(v: Any):
    """A key cell as a hashable value; ``None`` for a missing one (it never matches)."""
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            return None
        return int(f) if f.is_integer() else f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (str, np.str_)):
        return str(v) if str(v).strip() else None
    return v


def _row_keys(cols: Mapping[str, np.ndarray], keys: Sequence[str], n: int) -> List[Any]:
    vals = [np.asarray(cols[k]).tolist() for k in keys]
    out = []
    for i in range(n):
        parts = tuple(_key_value(v[i]) for v in vals)
        out.append(None if any(p is None for p in parts) else parts)
    return out


def _compute_table_join(ctx: EvalContext) -> Dataset:
    """A table with another table's columns added, matched row by row on a key.

    Resolved spec (V4.00 step 9, `build-node-v2`):

    * the LEFT table is ``table`` (of ``domain``) on ``data``; the OTHER is ``other_table``
      (of ``other_domain``) on the ``other`` input — or on ``data`` itself when ``other`` is
      unwired, to join two tables of one wire;
    * ``on`` picks the key: ``id_m_t`` (the same object in the same position and frame — two
      measurements of one segmentation), ``id`` (globally unique ids), ``m_t`` (a per-frame
      table onto every object of that frame), ``track`` (``track_id`` and ``t`` — a Track
      table's per-frame columns onto the tracked objects; untracked ``track_id`` 0 never
      matches). The batch member ``b`` joins the key when both tables carry it;
    * the OTHER table may hold each key once (else the match is ambiguous and is refused —
      aggregate it first); the left may repeat a key (many-to-one);
    * ``how`` left keeps every left row (unmatched: NaN numbers, empty text); inner keeps
      only the matched ones, in left order;
    * the other's columns are added as ``prefix`` + name — all but the key and its
      coordinates (``id, m, t, c, z, y, x``, ``b``), which only repeat the matched object; a
      name the left already has is refused rather than overwritten;
    * the output is ``data``'s Dataset plus the ``name`` table on the left's domain, with the
      left's 2D/3D kind; WHOLE_SERIES, tables only.
    """
    ds = ctx.inputs[0]
    other_in = ctx.input("other")
    other_ds = ds if other_in is None else other_in
    modes = ctx.params.get("__modes__", {}) or {}
    dom = TB.table_domain(modes)
    odom = TB.table_domain(modes, "other_domain")
    on = str(modes.get("on", "id_m_t"))
    how = str(modes.get("how", "left"))
    prefix = str(ctx.params.get("prefix", _DEFAULT_PREFIX) or "")
    name = ctx.layer("name") or _DEFAULT_NAME
    TB.refuse_voxel_name(ds, name, "Table Join")
    left_layer = _resolve_layer(
        _structure_layers(ds, dom), ctx.layer("table"), node="table join", socket="table",
        what=f"{dom.value} table", where="the `data` input",
        remedy=f"wire a node that makes a {dom.value} table, or switch Domain", ctx=ctx)[0]
    right_layer = _resolve_layer(
        _structure_layers(other_ds, odom), ctx.layer("other_table"), node="table join",
        socket="other_table", what=f"{odom.value} table",
        where="the `other` input" if other_in is not None else "the `data` input",
        remedy=f"wire a node that makes a {odom.value} table into Other, or switch Other "
               f"domain", ctx=ctx)[0]
    if other_in is None and odom is dom and name == right_layer and name != left_layer:
        raise ValueError(
            f"Table Join: Name {name!r} is the other table on this wire — the joined table "
            f"would replace it. Choose another Name (the default is {_DEFAULT_NAME!r}).")
    left = TB.read_table(ds, dom, left_layer)
    right = TB.read_table(other_ds, odom, right_layer)
    keys = list(_JOIN_KEYSETS.get(on, _JOIN_KEYSETS["id_m_t"]))
    if BATCH_COLUMN in left and BATCH_COLUMN in right:
        keys.append(BATCH_COLUMN)
    for side, cols, lyr in (("left", left, left_layer), ("other", right, right_layer)):
        miss = [k for k in keys if k not in cols]
        if miss:
            raise ValueError(
                f"Table Join: the {side} table {lyr!r} has no {miss} column(s) for On "
                f"{on!r} — it carries {sorted(cols)}. Pick another On.")
    nl = len(np.asarray(next(iter(left.values())))) if left else 0
    nr = len(np.asarray(next(iter(right.values())))) if right else 0
    rkeys = _row_keys(right, keys, nr)
    index: Dict[Any, int] = {}
    dup = 0
    for j, k in enumerate(rkeys):
        if k is None:
            continue
        if k in index:
            dup += 1
            continue
        index[k] = j
    if dup:
        raise ValueError(
            f"Table Join: {dup} row(s) of the other table {right_layer!r} repeat a key "
            f"({', '.join(keys)}), so a left row would match several — the join is "
            f"ambiguous. Aggregate the other table first (Table Aggregate), or join on a "
            f"finer key.")
    lkeys = _row_keys(left, keys, nl)
    match = np.array([index.get(k, -1) if k is not None and not (
        on == "track" and k[0] == 0) else -1 for k in lkeys], dtype=np.int64)
    rows = np.arange(nl) if how == "left" else np.flatnonzero(match >= 0)
    hit = match[rows]
    added = [c for c in right if c not in keys and c not in _JOIN_SKIP]
    clash = [prefix + c for c in added if prefix + c in left]
    if clash:
        raise ValueError(
            f"Table Join: the left table already has {clash} — set a Prefix for the other "
            f"table's columns (now {prefix!r}) so nothing is overwritten.")
    out: Dict[str, np.ndarray] = {c: np.asarray(v)[rows] for c, v in left.items()}
    ok = hit >= 0
    for c in added:
        src = np.asarray(right[c])
        if TB.is_text(src):
            vals = [str(src[j]) if j >= 0 else "" for j in hit.tolist()]
            out[prefix + c] = TB.text_column(vals)
        elif ok.all():
            out[prefix + c] = src[hit] if hit.size else src[:0]
        else:
            col = np.full(hit.size, np.nan)
            col[ok] = src[hit[ok]].astype(float)
            out[prefix + c] = col
    ctx.progress(1, 1, f"{int(ok.sum())} of {nl} rows matched on {', '.join(keys)} "
                       f"({how}); {len(added)} column(s) added")
    zk = ds.structure_zkind(dom, left_layer) or "plane_index"
    return TB.write_table(ds, dom, name, out, zk)


def _join_layers(params: Mapping[str, Any], modes: Mapping[str, Any]):
    return ((TB.table_domain(modes), TB.param_text(params, "name", _DEFAULT_NAME)),)


def _join_columns(params: Mapping[str, Any], modes: Mapping[str, Any],
                  incoming: Sequence = (), inputs: Sequence = ()):
    """The left table's columns, then the other table's non-key columns with the prefix
    (total). The other table's columns come from the ``other`` input's envelope."""
    try:
        dom = TB.table_domain(modes)
        odom = TB.table_domain(modes, "other_domain")
        name = TB.param_text(params, "name", _DEFAULT_NAME)
        raw = (params or {}).get("prefix", _DEFAULT_PREFIX)
        prefix = "" if raw is None else str(raw)
        envs = {sock: env for sock, env in (inputs or ())}
        left_env = envs.get("data")
        right_env = envs.get("other", left_env)
        if left_env is not None:
            lcols = TB.envelope_columns(
                left_env, dom, TB.envelope_layer(left_env, dom, TB.param_text(params, "table")))
        else:
            known = list(dict.fromkeys(lyr for d, lyr, _c in incoming or () if d is dom))
            want = TB.param_text(params, "table") or (known[0] if len(known) == 1 else "")
            lcols = [c for d, lyr, c in incoming or () if d is dom and lyr == want]
        rcols = [] if right_env is None else TB.envelope_columns(
            right_env, odom,
            TB.envelope_layer(right_env, odom, TB.param_text(params, "other_table")))
        keys = set(_JOIN_KEYSETS.get(str((modes or {}).get("on") or "id_m_t"),
                                     _JOIN_KEYSETS["id_m_t"])) | {BATCH_COLUMN}
        cols = list(lcols) + [prefix + c for c in rcols
                              if c not in keys and c not in _JOIN_SKIP]
        return tuple((dom, name, c) for c in dict.fromkeys(cols) if c)
    except Exception:                                   # noqa: BLE001 — total by contract
        return ()


_join_columns.wants_inputs = True


register_node(
    _compute_table_join, op_key="table.join", label="Table Join", category="table",
    inputs=[
        InDataset("data", description="The Dataset carrying the LEFT table — every row of "
                                       "it is kept (or only the matched ones, with How "
                                       "inner) — and passed on with the joined table."),
        InDataset("other", label="Other", passes_domains=False,
                  description="The Dataset carrying the table whose columns are added — "
                              "another branch's measurement, a tracking result. Leave it "
                              "unwired to join two tables of the `data` wire."),
        InString("table", "Table", field=False, default="", layer_in_mode="domain",
                 description="The left table. Blank uses the only one of its kind on the "
                             "wire, and asks when there are several."),
        InString("other_table", "Other table", field=False, default="",
                 layer_in_mode="other_domain", layer_from="other",
                 description="The table whose columns are added. Blank uses the only one of "
                             "its kind on the Other wire (or on `data` when Other is "
                             "unwired)."),
        InString("prefix", "Prefix", field=False, default=_DEFAULT_PREFIX,
                 description="Put before every added column's name (`other_area`), so the two "
                             "tables' columns stay apart. Blank keeps the names, and the node "
                             "refuses a name the left table already has."),
        InString("name", "Name", field=False, default=_DEFAULT_NAME,
                 description="Name of the joined table written on the output. The two input "
                             "tables are left as they are."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("domain", list(_JOIN_DOMAINS), default="label", label="Domain",
             description="Which kind of table the LEFT one is — the joined table is of the "
                         "same kind.",
             choice_docs=domain_docs(_JOIN_DOMAINS)),
        Mode("other_domain", list(_JOIN_DOMAINS), default="label", label="Other domain",
             description="Which kind of table the OTHER one is.",
             choice_docs=domain_docs(_JOIN_DOMAINS)),
        Mode("on", list(_JOIN_KEYSETS), default="id_m_t", label="On",
             description="Which columns must agree for two rows to match.",
             choice_docs={
                 "id_m_t": "The same object id in the same position and frame — two "
                           "measurements of ONE segmentation (another channel, a second "
                           "Measure). The safe default: per-plane ids repeat across frames.",
                 "id": "The object id alone — only for ids unique across the whole series "
                       "(3D or tracked labels); on per-frame ids it would match objects of "
                       "different frames.",
                 "m_t": "Position and frame — a per-frame table (one row per frame, e.g. a "
                        "background level) onto every object of that frame.",
                 "track": "track_id and frame — a Track table's per-frame columns onto the "
                          "tracked objects; an untracked object (track_id 0) never matches.",
             }),
        Mode("how", ["left", "inner"], default="left", label="How",
             description="What happens to a left row with no match.",
             choice_docs={
                 "left": "Keep every left row; where nothing matched the added columns are "
                         "empty (NaN, or blank text) — row counts stay those of the left.",
                 "inner": "Keep only the left rows that matched — the table shrinks to the "
                          "objects both tables know, which changes every count downstream.",
             }),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains_by_mode={"domain": {"label": frozenset({Domain.LABEL}),
                                      "point": frozenset({Domain.POINT}),
                                      "track": frozenset({Domain.TRACK})}},
    extra_layers=_join_layers, adds_columns=_join_columns,
    description="Add another table's columns to a Label, Point or Track table, matching rows "
                "on id+position+frame, id, position+frame or track — left or inner — as a "
                "new joined table.")
