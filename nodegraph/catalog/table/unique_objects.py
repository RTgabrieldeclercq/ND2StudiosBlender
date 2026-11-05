"""table.unique_objects — one physical object seen from several fields or frames: rows that
sit at the same stage position become one object, numbered by sighting (2026-10-13)."""
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
from nodegraph.registry import Granularity, InDataset, InFloat, InInt, InString, Mode, OutDataset

_UQ_DOMAINS = ("label", "point")
_DEFAULT_NAME = "unique"
_DEFAULT_ORDER = "t,m"
#: the columns this node adds to every row it passes on
UID_COLUMN = "object_uid"
SIGHTING_COLUMN = "sighting"
SIGHTINGS_COLUMN = "sightings"


def _uq_list(text: Any) -> List[str]:
    """A comma list as its distinct non-blank entries, in order (total)."""
    try:
        raw = "" if text is None else str(text)
    except Exception:                                   # noqa: BLE001 — total by contract
        return []
    return list(dict.fromkeys(s.strip() for s in raw.split(",") if s.strip()))


def _uq_components(xy: np.ndarray, tol: float, block: np.ndarray) -> np.ndarray:
    """Single-linkage components of the rows of ``xy`` closer than ``tol`` (Euclidean, the
    columns' own units), only ever linking rows of the same ``block`` (the batch member). A
    row with a non-finite coordinate is its own component. Returns a root index per row."""
    n = len(xy)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ok = np.all(np.isfinite(xy), axis=1)
    if tol > 0 and ok.sum() > 1:
        from scipy.spatial import cKDTree
        for b in np.unique(block[ok]):
            rows = np.flatnonzero(ok & (block == b))
            if len(rows) < 2:
                continue
            for i, j in cKDTree(xy[rows]).query_pairs(tol):
                ri, rj = find(int(rows[i])), find(int(rows[j]))
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


def _uq_order_key(cols: Mapping[str, np.ndarray], keys: Sequence[str], n: int) -> np.ndarray:
    """Each row's position in acquisition order as a float matrix (n, len(keys)) — a
    missing or non-numeric value sorts last."""
    out = np.full((n, max(len(keys), 1)), np.inf)
    for j, k in enumerate(keys):
        try:
            v = np.asarray(cols[k], dtype=float)
        except (TypeError, ValueError):
            continue
        out[:, j] = np.where(np.isfinite(v), v, np.inf)
    if not keys:
        out[:, 0] = 0.0
    return out


def _compute_unique_objects(ctx: EvalContext) -> Dataset:
    """Rows of one table that are the SAME physical object — seen from overlapping fields of
    view, or again in a later frame — get one ``object_uid`` and a ``sighting`` number.

    Resolved spec (2026-10-13, `build-node-v2`):

    * reads one table (``domain`` label / point, ``table``); its ``x`` / ``y`` columns
      (default ``stage_x_um`` / ``stage_y_um``, which Measure writes) place each row;
    * two rows closer than ``tolerance`` (in the x / y columns' own units — µm for the
      stage columns) are one object, by single linkage, so a chain of near rows is one
      object; rows of different batch members ``b`` are never linked; a row with no finite
      position is an object of its own; x and y only — z does not enter;
    * rows are put in acquisition order by ``order_by`` (comma columns, default ``t,m``:
      frame, then position — the order a multipoint time-lapse images them); within an
      object, rows with the same order key are ONE sighting (one frame saw it, e.g. split
      into two fragments), and ``sighting`` counts sightings 0, 1, 2, ...;
    * ``object_uid`` numbers the objects 1..K by their first sighting; ``sightings`` is
      each object's total, counted BEFORE ``keep_first`` drops any;
    * ``keep_first`` N > 0 keeps the rows of each object's first N sightings and drops the
      rest; 0 keeps every row (the columns are still added);
    * the output is a NEW table ``name`` of the same domain — the input table keeps every
      row, so its raster and overlays still agree with it; the kept rows keep every column
      and their ids, in the input's row order;
    * WHOLE_SERIES, tables only; no axis or calibration change.
    """
    ds = ctx.inputs[0]
    modes = ctx.params.get("__modes__", {}) or {}
    dom = TB.table_domain(modes)
    name = ctx.layer("name") or _DEFAULT_NAME
    TB.refuse_voxel_name(ds, name, "Unique Objects")
    layer = _resolve_layer(
        _structure_layers(ds, dom), ctx.layer("table"), node="unique objects",
        socket="table", what=f"{dom.value} table", where="the `data` input",
        remedy=f"wire a node that makes a {dom.value} table (Measure writes the stage "
               f"position columns), or switch Domain", ctx=ctx)[0]
    if name == layer:
        raise ValueError(
            f"Unique Objects: Name {name!r} is the table it reads — the filtered rows would "
            f"replace the ones its raster and overlays index. Choose another Name (the "
            f"default is {_DEFAULT_NAME!r}).")
    cols = TB.read_table(ds, dom, layer)
    xname = str(ctx.params.get("x", "stage_x_um") or "stage_x_um")
    yname = str(ctx.params.get("y", "stage_y_um") or "stage_y_um")
    missing = [c for c in (xname, yname) if c not in cols]
    if missing:
        raise ValueError(
            f"Unique Objects: the {layer!r} table has no {missing} column. The stage "
            f"position columns come from Measure (`stage_x_um`, `stage_y_um`, written when "
            f"the image carries a pixel size and a stage log); pick other X / Y columns "
            f"otherwise. The table carries {sorted(cols)}.")
    for c in (xname, yname):
        if TB.is_text(cols[c]):
            raise ValueError(f"Unique Objects: {c!r} holds text, not positions.")
    order = _uq_list(ctx.params.get("order_by", _DEFAULT_ORDER))
    unknown = [k for k in order if k not in cols]
    if unknown:
        raise ValueError(f"Unique Objects: Order by names {unknown}, which the {layer!r} "
                         f"table does not carry — it has {sorted(cols)}.")
    tol = float(ctx.params.get("tolerance", 3.0))
    if not math.isfinite(tol) or tol < 0:
        raise ValueError(f"Unique Objects: Tolerance must be a finite distance ≥ 0, got {tol}.")
    keep = int(ctx.params.get("keep_first", 0) or 0)
    if keep < 0:
        raise ValueError(f"Unique Objects: Keep first must be ≥ 0 (0 keeps every sighting), "
                         f"got {keep}.")

    xy = np.stack([np.asarray(cols[xname], dtype=float), np.asarray(cols[yname], dtype=float)],
                  axis=1)
    n = len(xy)
    block = (np.asarray(cols[BATCH_COLUMN]).astype(np.int64) if BATCH_COLUMN in cols
             else np.zeros(n, dtype=np.int64))
    roots = _uq_components(xy, tol, block)
    okey = _uq_order_key(cols, order, n)
    # rows in acquisition order (ties broken by row order, which keeps a frame's rows together)
    seq = np.lexsort(tuple(okey[:, j] for j in range(okey.shape[1] - 1, -1, -1)))
    uid = np.zeros(n, dtype=np.int64)
    sighting = np.zeros(n, dtype=np.int64)
    first: Dict[int, int] = {}
    last_key: Dict[int, Tuple] = {}
    count: Dict[int, int] = {}
    for i in seq.tolist():
        r = int(roots[i])
        k = tuple(okey[i].tolist())
        if r not in first:
            first[r] = len(first) + 1
            count[r] = 0
        elif k != last_key[r]:
            count[r] += 1
        last_key[r] = k
        uid[i] = first[r]
        sighting[i] = count[r]
    total = np.array([count[int(r)] + 1 for r in roots], dtype=np.int64)
    sel = np.ones(n, dtype=bool) if keep == 0 else sighting < keep
    out = {k: np.asarray(v)[sel] for k, v in cols.items()}
    out[UID_COLUMN] = uid[sel]
    out[SIGHTING_COLUMN] = sighting[sel]
    out[SIGHTINGS_COLUMN] = total[sel]
    k_obj = len(first)
    multi = int(sum(1 for r in first if count[r] > 0))
    ctx.progress(1, 1, f"{n} rows → {k_obj} object(s), {multi} seen more than once"
                       + (f"; kept {int(sel.sum())} rows (first {keep} sighting(s))"
                          if keep else ""))
    zk = ds.structure_zkind(dom, layer) or "plane_index"
    return TB.write_table(ds, dom, name, out, zk)


def _uq_layers(params: Mapping[str, Any], modes: Mapping[str, Any]):
    return ((TB.table_domain(modes), TB.param_text(params, "name", _DEFAULT_NAME)),)


def _uq_columns(params: Mapping[str, Any], modes: Mapping[str, Any],
                incoming: Sequence = (), inputs: Sequence = ()):
    """The read table's columns, then ``object_uid``, ``sighting`` and ``sightings``
    (total)."""
    try:
        dom = TB.table_domain(modes)
        name = TB.param_text(params, "name", _DEFAULT_NAME)
        envs = {sock: env for sock, env in (inputs or ())}
        env = envs.get("data")
        if env is not None:
            cols = TB.envelope_columns(
                env, dom, TB.envelope_layer(env, dom, TB.param_text(params, "table")))
        else:
            known = list(dict.fromkeys(lyr for d, lyr, _c in incoming or () if d is dom))
            want = TB.param_text(params, "table") or (known[0] if len(known) == 1 else "")
            cols = [c for d, lyr, c in incoming or () if d is dom and lyr == want]
        cols = list(cols) + [UID_COLUMN, SIGHTING_COLUMN, SIGHTINGS_COLUMN]
        return tuple((dom, name, c) for c in dict.fromkeys(cols) if c)
    except Exception:                                   # noqa: BLE001 — total by contract
        return ()


register_node(
    _compute_unique_objects, op_key="table.unique_objects", label="Unique Objects",
    category="table",
    inputs=[
        InDataset("data", description="The Dataset whose table is de-duplicated; it is passed "
                                       "on with the new table added and the input table "
                                       "unchanged."),
        InString("table", "Table", field=False, default="", layer_in_mode="domain",
                 description="Which table of the chosen kind holds the sightings. Blank uses "
                             "the only one on the wire, and asks when there are several."),
        InString("x", "X", field=False, default="stage_x_um", column_in_mode="domain",
                 column_from="table",
                 description="The column placing each row along x. The default "
                             "`stage_x_um` (from Measure) is the microscope stage position, "
                             "the same for one object in every field that sees it. A "
                             "per-field column like `x_um` would only match rows within one "
                             "field."),
        InString("y", "Y", field=False, default="stage_y_um", column_in_mode="domain",
                 column_from="table",
                 description="The column placing each row along y — `stage_y_um` by "
                             "default, for the same reason as X."),
        InFloat("tolerance", "Tolerance", unit="um", field=False, default=3.0,
                description="How close two rows must be to count as ONE object, in the X / Y "
                            "columns' units (µm for the stage columns). LARGER merges more — "
                            "two neighbouring objects closer than this become one. SMALLER "
                            "splits one object whose two fields disagree by more than this "
                            "(stage repeatability, a field's own centroid jitter) into two, "
                            "each with its own sightings. Keep it well under the objects' "
                            "spacing."),
        InString("order_by", "Order by", field=False, default=_DEFAULT_ORDER,
                 description="Comma-separated columns that put rows in acquisition order; "
                             "`t,m` (the default) is frame then position, the order a "
                             "multipoint time-lapse images them. Rows of one object with the "
                             "same values here are one sighting. Changes which sightings "
                             "`Keep first` keeps."),
        InInt("keep_first", "Keep first", field=False, default=0,
              description="Keep only each object's first N sightings, dropping its later "
                          "rows from the new table — e.g. 5 keeps the first five exposures of "
                          "an object a later field imaged again. 0 (the default) keeps every "
                          "row and only adds the columns. Every kept row's measurements are "
                          "unchanged; it changes how many rows a summary averages."),
        InString("name", "Name", field=False, default=_DEFAULT_NAME,
                 description="Name of the new table, of the same kind as the input. The "
                             "input table is left whole, so its raster and overlays still "
                             "match it."),
    ],
    outputs=[OutDataset()],
    modes=[
        Mode("domain", list(_UQ_DOMAINS), default="label", label="Domain",
             description="Which kind of table holds the sightings.",
             choice_docs=domain_docs(_UQ_DOMAINS)),
    ],
    granularity=Granularity.WHOLE_SERIES, kernel_axes=frozenset(),
    reads_domains_by_mode={"domain": {"label": frozenset({Domain.LABEL}),
                                      "point": frozenset({Domain.POINT})}},
    extra_layers=_uq_layers, adds_columns=_uq_columns,
    description="Find the rows of a Label or Point table that are the same physical object "
                "seen from overlapping fields or in later frames: each gets an "
                "`object_uid` and a `sighting` number in acquisition order, and Keep first "
                "drops the later sightings into a new table.")
